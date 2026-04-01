from typing import List
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from retrieval import RetrievalResult 
import json

llm_ollama = OllamaLLM(model = "mistral:latest")
groq_llm = ChatGroq(model="llama3-8b-8192")
gemini_flash_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
gemini_pro_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

QDRANT_SCORE_THRESHOLD = 0.3
POSTGRES_SCORE_THRESHOLD = 0.0


def rewrite_query(query: str, chat_history: List[str]|None) ->str:
    if not chat_history:
        return query
    
    prompt = f"""
        You are a query rewriter for an e-commerce product review assistant.
        Given the chat history and follow up question, rewrite the follow up as a fully self contained question.

        Chat History: {chat_history}
        Follow up Question: {query}
        
        Return only the rewritten question and nothing else.
    """
    
    rewritten_query = llm_ollama.invoke(prompt)
    return rewritten_query.strip()


def validate_retrieval(postgres_results: List[RetrievalResult]|None , qdrant_results:List[RetrievalResult]|None) -> dict:
    if not postgres_results and not qdrant_results:
        return {
            "valid": False,
            "reason": "Both sparse and dense retrieval returned no results"
        }
    
    postgres_valid = False
    if postgres_results:
        valid_rows = [
            row for row in postgres_results if row.text.strip() != "" and row.score > POSTGRES_SCORE_THRESHOLD and row.asin_id is not None and row.metadata.get("title", "").strip() != ""
        ]
        postgres_valid = len(valid_rows) >= 1
    
    qdrant_valid = False
    if qdrant_results:
        valid_reviews = [
            row for row in qdrant_results if row.score >= QDRANT_SCORE_THRESHOLD and row.text is not None and row.text.strip() != ""and row.asin_id is not None
        ]
        qdrant_valid = len(valid_reviews) >= 1

    if not postgres_valid and not qdrant_valid:
        return {
            "valid": False,
            "reason": "Results returned but failed quality checks for both sparse and dense retrieval"
        }
    
    if not postgres_valid:
        return {
            "valid": True,
            "reason": f"Only dense results passed quality check, {len(qdrant_results)} reviews found"
        }
    
    if not qdrant_valid:
        return {
            "valid": True,
            "reason": f"Only sparse results passed quality check, {len(postgres_results)} products found"
        }
    
    return {
        "valid": True,
        "reason": f"Both passed => {len(postgres_results)} products, {len(qdrant_results)} reviews"
    }


def build_prompt(query:str, postgres_results: List[RetrievalResult]|None, qdrant_results: List[RetrievalResult]|None) -> str:
    prompt =f"""
        You are an assistant of e-commerce product review analysis.
        Your task is to analyze user queries to provide accurate and helpful answers based on retrieved product or review information.
        You will be provided with a user query and relevant information from product database(sparse retrieval) or customer reviews (dense retrieval) or both.

        User Query: {query}
    """

    if postgres_results:
        prompt += "\n [PRODUCT FACTS] (from sparse retrieval):\n"
        for item in postgres_results:
            title = item.metadata.get("title", "Unknown")
            brand = item.metadata.get("brand", "Unknown")
            category = item.metadata.get("category", "Unknown")
            price = item.metadata.get("price", "Unknown")
            price_raw = item.metadata.get("price_raw", "Unknown")
            prompt += f" Product: {title} | Brand: {brand} | Category: {category} | Price: {price} ({price_raw}) | Score: {item.score:.2f}\n"

    if qdrant_results:
        prompt += "\n [CUSTOMER REVIEWS] (from dense retrieval):\n"
        for item in qdrant_results:
            asin = item.metadata.get("asin")
            if not asin:
                continue
            prompt += f"- Product {asin}: {item.text} | Relevance: {item.score:.2f}\n"

    prompt += """
        [INSTRUCTIONS]
        - Answer concisely and directly
        - Cite whether your answer comes from product facts or customer reviews
        - If asked for opinion, use customer reviews
        - If asked for specs or price, use product facts
        - Based on the information provided, answer the user's query. If information is insufficient to provide confident answer, say you don't know.
        - Do not make up information not present above
    """
    return prompt


def generate_answer(prompt:str, model_number:int) -> str:
    #Escalation Ladder
    if model_number == 1:
        llm = gemini_flash_llm
    elif model_number == 2:
        llm = gemini_pro_llm
    else:
        llm = groq_llm

    answer = llm.invoke(prompt)

    return answer.content.strip()


def evaluate_answer(question: str, answer: str, context: str) -> dict:
    
    prompt = f"""
    You are an evaluator for an e-commerce product review assistant.
    Given a question, an answer, and the context used to generate the answer, evaluate the quality of the answer.
    
    QUESTION: {question}
    CONTEXT:{context}
    ANSWER:{answer}

    Evaluate the answer and respond in JSON only with this exact format:
    {{
        "score": <float between 0.0 and 1.0>,
        "failure_type": "<hallucination|incomplete|irrelevant|refusal|none>",
        "is_refusal": <true|false>,
        "root_cause": "<missing_entity_coverage|keyword_miss|semantic_mismatch|none>"
    }}

    Scoring guide:
    - 1.0: answer is fully grounded in context and directly answers the question
    - 0.7-0.9: answer is mostly correct but missing some details
    - 0.4-0.6: answer is partially correct or vague
    - 0.0-0.3: answer is wrong, hallucinated, or completely irrelevant

    Failure types:
    - hallucination: answer contains information not present in context
    - incomplete: answer is too vague or missing key details
    - irrelevant: answer does not address the question
    - refusal: answer says it does not have enough information
    - none: answer is good

    Root cause definitions:
    - missing_entity_coverage: relevant entities/products are missing from context
    - keyword_miss: sparse retrieval failed due to missing keywords
    - semantic_mismatch: dense retrieval returned semantically irrelevant results
    - none: no clear root cause
    Return JSON only, no explanation, no markdown backticks."""

    raw = groq_llm.invoke(prompt)
    
    try:
        result = json.loads(raw.content.strip())
        return {
            "score": float(result.get("score", 0.0)),
            "failure_type": result.get("failure_type", "none"),
            "is_refusal": bool(result.get("is_refusal", False)),
            "root_cause": result.get("root_cause", "none")
        }
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "failure_type": "incomplete",
            "is_refusal": False,
            "root_cause": "none"
        }