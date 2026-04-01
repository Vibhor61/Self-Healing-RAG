import pandas as pd
import os
from dataclasses import dataclass
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import psycopg2
from typing import List, Optional
from copy import deepcopy
import enum

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

POSTGRES_SCORE_THRESHOLD = 0.01
QDRANT_SCORE_THRESHOLD = 0.5


class RetrievalStatus(enum.Enum):
    Passed = "passed"
    empty = "empty"
    low_quality = "low_quality"
    semantic_failure = "semantic_failure"
    keyword_miss = "keyword_miss"


class QueryType(enum.Enum):
    
@dataclass
class RetrievalSignals:
    sparse_hit_count: int
    avg_sparse_score: float        
    top_sparse_score: float

    dense_hit_count: int
    avg_dense_score: float       
    top_dense_score: float

    asin_overlap: int
    mode_used: str


@dataclass 
class RetrievalResult:
    source : str
    doc_id : str
    review_id: Optional[int]
    asin_id: str
    text: str
    score: float
    rank : int
    metadata: dict


@dataclass
class FinalResult:
    query: str
    resolved_asin: Optional[str]
    items: List[RetrievalResult]

    status: RetrievalStatus = RetrievalStatus.Passed
    failure_reason: Optional[str] = None
    signals: Optional[RetrievalSignals] = None


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def compute_retrieval_signals(sparse_results: List[RetrievalResult], dense_results: List[RetrievalResult], mode_used):
    sparse_scores = [r.score for r in sparse_results]
    dense_scores = [r.score for r in dense_results]
    
    sparse_hit_count=len(sparse_results),
    top_sparse_score=max(sparse_scores) if sparse_scores else 0.0,
    avg_sparse_score=sum(sparse_scores) / len(sparse_scores) if sparse_scores else 0.0,

    dense_hit_count=len(dense_results),
    top_dense_score=max(dense_scores) if dense_scores else 0.0,
    avg_dense_score=sum(dense_scores) / len(dense_scores) if dense_scores else 0.0,
    
    min_overlap = compute_asin_overlap(sparse_results, dense_results)

    return RetrievalSignals(
        sparse_hit_count = sparse_hit_count,
        avg_sparse_score = avg_sparse_score,
        top_sparse_score = top_sparse_score,

        dense_hit_count = dense_hit_count,
        avg_dense_score = avg_dense_score,
        top_dense_score = top_dense_score,
        asin_overlap = min_overlap,
        mode_used = mode_used
    )

def compute_asin_overlap(sparse_results, dense_results):
    if not sparse_results or not dense_results:
        return 0
    
    sparse_asins = set([r.asin_id for r in sparse_results])
    dense_asins = set([r.asin_id for r in dense_results])
    return len(sparse_asins & dense_asins)


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
            doc_id=str(item.id),
            review_id=item.payload.get("review_id"),
            asin_id=item.payload.get("asin"),
            text=item.payload.get("text"),
            score=item.score,
            rank=rank,
            metadata={"review_id": item.payload.get("review_id"), "asin": item.payload.get("asin")}
        ))
    return retrieval_results
    

def compute_stats(sparse_results, dense_results):
    if not sparse_results and not dense_results:
        return {"sparse_count": 0, "dense_count": 0, "overlap_count": 0, "avg_sparse_score": 0, "avg_dense_score": 0}
    
    if not sparse_results:
        dense_asins = set([r.asin_id for r in dense_results])
        return {
            "sparse_hits": 0,
            "dense_hits": len(dense_results),
            "overlap": 0,
            "avg_sparse_score":0,
            "avg_dense_score": sum([r.score for r in dense_results]) / (len(dense_results) or 1),
        }

    if not dense_results:
        sparse_asins = set([r.asin_id for r in sparse_results])
        return {
            "sparse_hits": len(sparse_results),
            "dense_hits": 0,
            "overlap": 0,
            "avg_sparse_score": sum([r.score for r in sparse_results]) / (len(sparse_results) or 1),
            "avg_dense_score": 0,
        }
    sparse_asins = set([r.asin_id for r in sparse_results])
    dense_asins = set([r.asin_id for r in dense_results])
    
    return {
        "sparse_hits": len(sparse_results),
        "dense_hits": len(dense_results),
        "overlap": len(sparse_asins & dense_asins),
        "avg_sparse_score": sum([r.score for r in sparse_results]) / (len(sparse_results) or 1),
        "avg_dense_score": sum([r.score for r in dense_results]) / (len(dense_results) or 1),
    }


def compute_rrf_score(item, fusion_k=60, w_sparse=0.5, w_dense=0.5):
    if item.rank is None:
        raise ValueError("Rank cannot be None for RRF score computation")
    
    if item.source == "sparse":
        weight = w_sparse
    else:
        weight = w_dense

    return weight * (1.0 / (fusion_k + item.rank))


def fusion_retrieval(query: str, top_k: int = 5, k :int =60) -> FinalResult:
    sparse_results = sparse_fact_retrieval(query, top_k)
    dense_results = dense_fact_retrieval(query, top_k)

    scores = {}
    best_asin = {}

    for item in sparse_results + dense_results:
        if item.rank is None:
            raise ValueError("Rank cannot be None")

        key = f"{item.source}:{item.doc_id}"
        
        rrf_score = compute_rrf_score(item, fusion_k=k, w_sparse=0.5, w_dense=0.5)
        scores[key] = scores.get(key, 0) + rrf_score

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

    stats = compute_stats(sparse_results, dense_results)

    return FinalResult(
        query=query,
        resolved_asin=fused_results[0].asin_id if fused_results else None,
        items=fused_results,
        retrieval_stats=stats
    )