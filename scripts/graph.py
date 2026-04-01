from langgraph.graph import StateGraph, END
from typing import List, Optional, Annotated, TypedDict
import operator
from router import route
from retrieval import sparse_fact_retrieval, dense_fact_retrieval, fusion_retrieval
from answer import build_prompt, rewrite_query,validate_retrieval, generate_answer, evaluate_answer
import phoenix as px
from phoenix.trace import trace

px.launch_app()

class State(TypedDict):
    history: Annotated[list[str], operator.add]
    current_query: str

    routing_decision: str
    routing_confidence: float

    retrieval_type: str
    retrieved_data: List[dict] 
    retrieval_score: float
    retrieval_retries: int
    retrieval_valid: bool

    answer: str
    model_used: str
    answer_score: float
    retries: int

    error_message: Optional[str]
    user_satisfied: Optional[bool]
    should_clarify: Optional[bool]

@trace
def router_node(state: State):
    result = route(state["current_query"])
    return {
        "routing_decision": result.retrieval_type,
        "routing_confidence": result.confidence
    }

@trace
def retrieve_node(state: State):
    query = state["current_query"]
    decision = state["routing_decision"]
    if decision == "sparse":
        retrieval_results = sparse_fact_retrieval(query)
    elif decision == "dense":
        retrieval_results = dense_fact_retrieval(query)
    else:
        retrieval_results = fusion_retrieval(query)

    return {
        "retrieval_type": decision,
        "retrieved_data": [item.dict() for item in retrieval_results],
        "retrieval_score": retrieval_results.items.score,
    }

@trace
def validate_node(state: State):
    postgres_results = False
    qdrant_results = False
    if state["retrieval_type"] == "dense": 
        qdrant_results = True
    if state["retrieval_type"] == "sparse":
        postgres_results = True
    if state["retrieval_type"] == "fusion":
        qdrant_results = True
        postgres_results = True

    check = validate_retrieval(postgres_results=postgres_results, qdrant_results=qdrant_results)
    return {
        "retrieval_valid": check["valid"]
    }

@trace
def generate_node(state: State):
    retrieved_items = state["retrieved_data"]
    prompt = build_prompt(state["current_query"], retrieved_items)
    answer, model_used = generate_answer(prompt)
    return {
        "answer": answer,
        "model_used": model_used
    }

@trace
def evaluate_node(state: State):
    #Will add RAGAS evaluation in future along with this
    evaluation_score = evaluate_answer(state["current_query"], state["answer"])
    return {
        "answer_score": evaluation_score
    }

# CONDITIONAL NODES 
def router_guard(state: State):
    if state["routing_confidence"] < 0.5:
        return "clarify"
    return "retrieve"


def retrieval_guard(state: State):
    if not state["retrieval_valid"]:
        if state["retrieval_retries"] >= 2:
            return "end"
        return "retrieve"
    return "generate"


def generation_guard(state: State):
    if state["answer_score"] < 0.7:
        if state["retries"] >= 2:
            return "end"
        return "generate"
    return "end"


graph = StateGraph(initial_state=State())

graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("validate", validate_node)
graph.add_node("generate", generate_node)
graph.add_node("evaluate", evaluate_node)

graph.add_node("rewrite", lambda s: {"current_query": rewrite_query(s["current_query"], s["history"]), "retrieval_retries": s["retrieval_retries"]+1})
graph.add_node("escalate", lambda s: {"error_message": "Escalating to better model", "model_index": s["model_index"]})


graph.set_entry_point("router")
graph.add_conditional_edge("router", router_guard,{
    "clarify": "rewrite",
    "retrieve": "retrieve"
})

graph.add_edge("retrieve", "validate")

graph.add_conditional_edge("validate", retrieval_guard, {
    "retrieve": "retrieve",
    "generate": "generate",
    "end": END
})

graph.add_edge("generate", "evaluate")

graph.add_conditional_edge("evaluate", generation_guard, {
    "generate": "generate",
    "end": END
})

graph.add_edge("rewrite", "retrieve")