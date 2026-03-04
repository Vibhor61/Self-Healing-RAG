import pandas as pd
import os
from dataclasses import dataclass
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import psycopg2
from typing import List, Optional
from copy import deepcopy

DB_CONFIG = {
    "host" : os.getenv("POSTGRES_HOST"),
    "database" : os.getenv("POSTGRES_DB"),
    "user" : os.getenv("POSTGRES_USER"),
    "password" : os.getenv("POSTGRES_PASSWORD"),
    "port" : int(os.getenv("POSTGRES_PORT"))
}

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

@dataclass 
class RetrievalResult:
    source : str
    doc_id : str
    review_id: Optional[int]
    asin_id: str
    text: str
    score: float
    rank : str
    metadata: dict

@dataclass
class FinalResult:
    query: str
    resolved_asin: Optional[str]
    items: List[RetrievalResult]

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def sparse_fact_retrieval(query: str, top_k: int = 5) -> List[RetrievalResult]:
    conn = get_connection()
    cursor = conn.cursor()
    
    sql_query = """
        SELECT asin, title, brand, category, price, price_raw,
            ts_rank_cd(search_vector, websearch_to_tsquery('english', %s)) AS score
        FROM products_table
        WHERE search_vector @@ websearch_to_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s;"""
    
    cursor.execute(sql_query, (query, query, top_k))
    results = cursor.fetchall() 
    retrieval_results = []
    for rank, row in enumerate(results):
        retrieval_results.append(RetrievalResult(
            source="sparse",
            doc_id=row[0],
            review_id=None,
            asin_id=row[0],
            text=f"{row[1]} {row[2]} {row[3]} {row[4]} {row[5]}",
            score=row[6],
            rank=rank,
            metadata={"title": row[1], "brand": row[2], "category": row[3], "price": row[4], "price_raw": row[5]}
        ))
    cursor.close()
    conn.close()
    return retrieval_results

def dense_fact_retrieval(query: str, top_k: int = 5) -> List[RetrievalResult]:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    query_embedding = EMBEDDING_MODEL.encode(query).tolist()
    
    search_result = client.search(
        collection_name="reviews_embeddings",
        query_vector=query_embedding,
        limit=top_k
    )
    
    retrieval_results = []
    for rank, item in enumerate(search_result):
        retrieval_results.append(RetrievalResult(
            source="dense",
            doc_id=item.id,
            review_id=item.payload.get("review_id"),
            asin_id=item.payload.get("asin"),
            text=item.payload.get("text"),
            score=item.score,
            rank=rank,
            metadata={"review_id": item.payload.get("review_id"), "asin": item.payload.get("asin")}
        ))
    return retrieval_results
    
def fusion_retrieval(query: str, top_k: int = 5, k :int =60) -> FinalResult:
    sparse_results = sparse_fact_retrieval(query, top_k)
    dense_results = dense_fact_retrieval(query, top_k)

    scores = {}
    best_asin = {}

    for sparse,dense in zip(sparse_results, dense_results):
        for item in [sparse]+[dense]:
            if item.rank is None:
                raise ValueError("Rank cannot be None")
            
            key = f"{item.source}:{item.doc_id}"
            scores[key] = scores.get(key, 0) + 1.0/k+item.score
            if key not in best_asin or best_asin[key].rank > item.rank:
                best_asin[key] = item

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]        
    
    fused_results = []
    for final_rank, (key, score) in enumerate(ordered, start=1):
        base = best_asin[key]
        copied = deepcopy(base)

        copied.rank = final_rank
        copied.score = score

        copied.metadata = dict(copied.metadata or {})
        copied.metadata.update({
            "rrf_k": k,
            "rrf_score": float(score),
            "original_source": base.source,
            "original_rank": base.rank,
        })

        fused_results.append(copied)