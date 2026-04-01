import json
from typing import Optional
from dataclasses import dataclass
from router import RouterResult
from langchain_ollama import OllamaLLM

@dataclass
class PlanResult:
    mode: str
    top_k: int

    weights: Optional[dict] = None
    fusion_k: Optional[int] = None


def planner_rewriter(query: str, failure_type: str, root_cause: str, iteration: int) -> str:
    if iteration >= 2:
        return query
    
    if root_cause == "missing_entity_coverage":
        return query + " specifications features details reviews"

    if root_cause == "keyword_miss":
        return query + " product name brand model alternatives"

    if root_cause == "semantic_mismatch":
        tokens = query.split()
        return " ".join(tokens[: min(len(tokens), 6)])

    if failure_type == "incomplete":
        return query + " more details"

    if failure_type == "irrelevant":
        return query + " correct product context"

    if failure_type == "hallucination":
        return query + " exact specification"

    return query


def initial_iteration(router_result: RouterResult) -> PlanResult:
    mode = router_result.retrieval_type
    if not mode:
        mode = "hybrid"
    signals = router_result.signals or {}
    if signals:
        signal_type = signals.get("type")

    
    if mode == "sparse":
        # Need less top_k for sparse retrieval
        top_k = 5
    elif mode == "dense":
        top_k = 8
    else:
        # Default values
        top_k = 12
        weights = {"sparse":0.5, "dense":0.5}
        fusion_k = 60
    
    if signal_type == "exact_match":
        top_k = 3
    elif signal_type == "fuzzy_match":
        top_k = 6
    elif signal_type == "keyword_conflict":
        top_k += 2
    elif signals.get("llm_fallback"):
        top_k += 3

    top_k = max(3, min(top_k, 15))

    return PlanResult(
        mode=mode,
        top_k=top_k,
        fusion_k=fusion_k,
        weights=weights
    )


def llm_based_fallback(llm_type:int, router_result: RouterResult, prev_plan: PlanResult, evaluation: dict, query:str) -> PlanResult:
    if llm_type == 1:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    elif llm_type == 2:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
    prompt = f"""
        You are a retrieval planner for an e-commerce product review assistant. 
        Based on the initial retrieval plan, retrieval results and their evaluation, create a new retrieval plan for the next iteration. 

        Query: {query}
        Initial Retrieval Plan: {prev_plan}
        Retrieval Results Evaluation: {evaluation}
        Router Signals: {router_result.signals}

        Consider the following when creating the new plan:
        - If the evaluation indicates that the retrieved results are mostly irrelevant, increase top_k and consider switching to hybrid mode if not already in it.
        - If there are signals indicating a strong intent (e.g., exact match), prioritize that retrieval type and adjust top_k accordingly.
        - If there are conflicting signals, try to balance them by adjusting weights or increasing top_k to allow for more results to be considered.

        Return the new retrieval plan in JSON format with fields: mode, top_k, weights (if hybrid), and fusion_k (if hybrid).
    """

    response = llm.invoke(prompt)
    try:
        plan_dict = json.loads(response)
        return PlanResult(
            mode=plan_dict.get("mode", response.mode),
            top_k=plan_dict.get("top_k", response.top_k),
            weights=plan_dict.get("weights", response.weights),
            fusion_k=plan_dict.get("fusion_k", response.fusion_k)
        )
    except json.JSONDecodeError:
        return None
    

def adaptive_iteration(query: str, router_result: RouterResult, iteration: int, prev_plan: PlanResult, evaluation: dict) -> PlanResult:
    if iteration > 3:
        return PlanResult(mode="ask_for_human_clarification", top_k=0, weights=None, fusion_k=None)
    
    score = evaluation["score"]
    failure = evaluation["failure_type"]
    is_refusal = evaluation["is_refusal"]

    mode = prev_plan.mode
    top_k = prev_plan.top_k
    weights = prev_plan.weights
    fusion_k = prev_plan.fusion_k

    handled = False
    if is_refusal:
        top_k += 5

    elif failure == "incomplete":
        top_k += 3

    elif failure == "hallucination":
        mode = "hybrid"
        top_k += 2
        weights = {"sparse": 0.7, "dense": 0.3}

    elif failure == "irrelevant":
        mode = "hybrid"
        weights = {"sparse": 0.4, "dense": 0.6}

    elif failure == "none" and score < 0.7:
        handled = True

    else:
        handled = True

    
    if iteration == 0:
        return initial_iteration(router_result)
    else:
        return llm_based_fallback(llm_type=iteration, router_result=router_result, prev_plan=prev_plan, evaluation=evaluation)
    

def planner



