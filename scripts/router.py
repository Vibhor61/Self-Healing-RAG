import json
from rapidfuzz import process,fuzz
import re
import os
import psycopg2
from dataclasses import dataclass
from langgraph import OllamaLLM
import enum
from typing import Optional, Set


DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST","postgres"),
    "database": os.getenv("POSTGRES_DB","rag_db"),
    "user": os.getenv("POSTGRES_USER","rag_user"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": int(os.getenv("POSTGRES_PORT",5432))
}

FUZZY_THRESHOLD = 90
MAX_PHRASE_LENGTH = 4
MIN_PHRASE_CHARS = 3 

SPARSE_KEYWORDS = {"price", "cost", "brand", "category", "cheap", "expensive", "how much", "rate"}
DENSE_KEYWORDS = {"best", "comfortable", "recommend", "review", "good", "worst", "experience", "quality", "feel", "worth"}
HYBRID_KEYWORDS = {"compare", "vs", "versus", "difference", "better", "between"}

KNOWN_CATEGORIES = set()
KNOWN_BRANDS = set()

class RouterStatus(enum.Enum):
    Passed = "passed"
    Low_Confidence = "low_confidence"
    Vague = "vague"
    Ambiguous = "ambiguous"
    LLM_fallback = "llm_fallback"
    Failed = "failed"


class QueryType(enum.Enum):
    PRODUCT_LOOKUP = "product_lookup"   
    REVIEW_QUERY = "review_query"     
    COMPARISON = "comparison"           
    VAGUE = "vague"                     


@dataclass
class RouterResult:
    retrieval_type: str
    query_type: QueryType
    status: RouterStatus

    reason: str
    signals: dict
    llm_used: bool 
    
    entity_signal: Optional[str] = None
    failure_reason: Optional[str] = None


def load_data():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT brand, category FROM products_table")
    
    for  brand, category in cur.fetchall():
        KNOWN_CATEGORIES.add(category)
        KNOWN_BRANDS.add(brand)
    cur.close()
    conn.close()


def preprocess_query(query:str) -> list[str]:
    query = query.lower()
    query = re.sub(r'[^\w\s]', ' ', query)
    return query.split()


# It will be used for isolating query type for interpretability after Evaluation
def map_query_type(retrieval_type:str, tokens: Set[str]) -> QueryType:
    if tokens & HYBRID_KEYWORDS:
        return QueryType.COMPARISON
    if tokens & DENSE_KEYWORDS:
        return QueryType.REVIEW_QUERY
    if tokens & SPARSE_KEYWORDS:
        return QueryType.PRODUCT_LOOKUP
    

    if retrieval_type == "sparse":
        return QueryType.PRODUCT_LOOKUP
    if retrieval_type == "dense":
        return QueryType.REVIEW_QUERY
    if retrieval_type == "hybrid":
        return QueryType.REVIEW_QUERY
    
    return QueryType.VAGUE
    

def check_phrases(query: str) -> str | None:
    query_lower = query.lower()

    for phrase in HYBRID_KEYWORDS:
        if " " in phrase and phrase in query_lower:
            return "hybrid"
 
    sparse_hit = any(phrase in query_lower for phrase in SPARSE_KEYWORDS if " " in phrase)
    dense_hit = any(phrase in query_lower for phrase in DENSE_KEYWORDS if " " in phrase)
 
    if sparse_hit and dense_hit:
        return "hybrid"
    if sparse_hit:
        return "sparse"
    if dense_hit:
        return "dense"
 
    return None


def exact_match(query:str) -> tuple[str|None, str|None]:
    tokens = preprocess_query(query)
    nums_tokens = len(tokens)

    for n in range(MAX_PHRASE_LENGTH, 0, -1):
        for i in range(nums_tokens - n + 1):
            phrase = " ".join(tokens[i:i+n])
            
            if phrase in KNOWN_BRANDS:
                return "brand", phrase
            if phrase in KNOWN_CATEGORIES:
                return "category", phrase
    return None, None
        

def fuzzy_match(query:str) -> tuple[str|None, str|None, float]:
    tokens = preprocess_query(query)
    
    best_brand_value, best_brand_score = None, 0
    best_category_value, best_category_score = None, 0

    phrases = []
    for i in range(1,MAX_PHRASE_LENGTH+1):
        for j in range(len(tokens)-i+1):
            phrase = " ".join(tokens[j:j+i])
            if len(phrase) >= MIN_PHRASE_CHARS:    
                phrases.append(phrase)
    
    # For simplicity, we take the best match across all tokens for brands and categories
    for phrase in phrases:
        if KNOWN_BRANDS:
            brand_match, brand_score, _ = process.extractOne(phrase, KNOWN_BRANDS, scorer=fuzz.token_sort_ratio) 
            if brand_match and brand_score > best_brand_score:
                best_brand_value = brand_match
                best_brand_score = brand_score

        if KNOWN_CATEGORIES:
            category_match, category_score, _= process.extractOne(phrase, KNOWN_CATEGORIES, scorer=fuzz.token_sort_ratio) 
            if category_match and category_score > best_category_score:
                best_category_value = category_match
                best_category_score = category_score
    
    if best_brand_value and best_brand_score >= FUZZY_THRESHOLD and best_category_score >= FUZZY_THRESHOLD:
        return "both", f"{best_brand_value},{best_category_value}"
   
    if best_brand_score >= FUZZY_THRESHOLD:
        return "brand", best_brand_value
    
    if best_category_score >= FUZZY_THRESHOLD:
        return "category", best_category_value
    
    return None, None
            

def llm_fallback(query:str) -> RouterResult:
    ollama = OllamaLLM(model="mistral:latest")
    try:
        ollama_response = f"""
            You are a query classifier for an e-commerce product review assistant.
            Your task is to classify retrieval strategy for user queries into three categories: 
                1) sparse: for factual queries about price, brand, category, specifications
                2) dense: for opinion based queries about recommendations, reviews, experiences  
                3) hybrid: for queries that need both facts and opinions
                        
            Response with json only with the following format:
                {
                    "retrieval_type": "sparse|dense|hybrid",
                    "reason": "explanation of why this retrieval type was chosen"
                }
            Classify the intent of following {query} and respond with the appropriate retrieval type and reason.
        """
        
        raw = ollama.invoke(prompt = ollama_response)
        result = json.loads(raw.content.strip())

        retrieval_type = result.get("retrieval_type")
        query_type = map_query_type(retrieval_type, set(preprocess_query(query)))
        return RouterResult(
                retrieval_type=retrieval_type,
                query_type=query_type,
                status=RouterStatus.LLM_fallback,
                reason=result.get("reason", "LLM classification"),
                signals={"type": "llm_fallback"},
                llm_used=True
            )
    
    except Exception as e:
        return RouterResult(
            retrieval_type="hybrid",
            query_type=QueryType.VAGUE,
            status=RouterStatus.Failed,
            reason="LLM fallback failed",
            signals={"type": "llm_error"},
            llm_used=True,
            failure_reason=str(e)
        )


def route(query:str) -> RouterResult:
    tokens = set(preprocess_query(query))
    phrase_result = check_phrases(query)
    if phrase_result:
        return RouterResult(
            retrieval_type=phrase_result,
            query_type=map_query_type(phrase_result, tokens),
            status=RouterStatus.Passed,
            reason="Phrase match detected",
            signals={"type": "phrase_match"},
            llm_used=False
        )
    

    if tokens & HYBRID_KEYWORDS:
        matched = tokens & HYBRID_KEYWORDS
        return RouterResult(
            retrieval_type="hybrid",
            query_type=QueryType.COMPARISON,
            status=RouterStatus.Passed,
            reason=f"Hybrid keywords: {matched}",
            signals={"type": "keyword","matched": list(matched)},
            llm_used=False
        )
    
    sparse_hits = tokens & SPARSE_KEYWORDS
    dense_hits = tokens & DENSE_KEYWORDS

    if sparse_hits and not dense_hits:
        return RouterResult(
           retrieval_type="sparse",
            query_type=QueryType.PRODUCT_LOOKUP,
            status=RouterStatus.Passed,
            reason="Sparse keywords",
            signals={"type": "keyword"},
            llm_used=False
        )

    if dense_hits and not sparse_hits:
        return RouterResult(
            retrieval_type="dense",
            query_type=QueryType.REVIEW_QUERY,
            status=RouterStatus.Passed,
            reason="Dense keywords",
            signals={"type": "keyword"},
            llm_used=False
        )

    if sparse_hits and dense_hits:
        return RouterResult(
            retrieval_type="hybrid",
            query_type=QueryType.REVIEW_QUERY,  
            status=RouterStatus.Ambiguous,
            reason="Conflicting signals",
            signals={"type": "keyword"},
            llm_used=False
        )
    
   
    match_type, match_value = exact_match(query)
    if match_type is not None:
        return RouterResult(
            retrieval_type="sparse",
            query_type=QueryType.PRODUCT_LOOKUP,
            status=RouterStatus.Passed,
            reason="Exact match",
            signals={"type": "exact_match"},
            llm_used=False,
            entity_signal=match_value
        )
   
    
    match_type_fuzzy, match_value, confidence = fuzzy_match(query)
    if match_type_fuzzy is not None:
        if match_type_fuzzy == "both":
            return RouterResult(
                retrieval_type="hybrid",
                query_type=QueryType.REVIEW_QUERY,
                status=RouterStatus.Low_Confidence,
                reason=f"Fuzzy match (brand + category): '{match_value}'",
                signals={"type": "fuzzy_match", "field": "both", "value": match_value},
                llm_used=False,
                confidence=confidence,
                entity_signal=match_value
            )

        return RouterResult(
            retrieval_type="sparse",
            query_type=QueryType.PRODUCT_LOOKUP,
            status=RouterStatus.Low_Confidence,
            reason=f"Fuzzy {match_type_fuzzy} match: '{match_value}'",
            signals={"type": "fuzzy_match", "field": match_type_fuzzy, "value": match_value},
            llm_used=False,
            entity_signal=match_value
        )
    
    if len(tokens) <= 2:
        return RouterResult(
            retrieval_type="none",
            query_type=QueryType.VAGUE,
            status=RouterStatus.Vague,
            reason="Too vague",
            signals={"type": "vague"},
            llm_used=False,
            failure_reason="insufficient signal"
        )

    return llm_fallback(query)

load_data()