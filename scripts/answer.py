from typing import List, Optional, TypedDict,Annotated
import os
from langgraph.graph import StateGraph
from retrieval import sparse_fact_retrieval, dense_fact_retrieval, FinalResult, RetrievalResult, fusion_retrieval
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

class state(TypedDict):
    query: str
    resolved_asin: Optional[str]
    retrieved_items: List[dict]
    answer: str

def products_node(state):
    query = state["query"]
    sparse_results = sparse_fact_retrieval(query)
    


