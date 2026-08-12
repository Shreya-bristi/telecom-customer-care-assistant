"""
Run a LangSmith evaluation of the RAG chain.

Usage:
    python -m telecom_rag.evaluation.run_eval

This script:
  1. Loads (or creates) the eval dataset in LangSmith.
  2. Runs the RAG chain against each example.
  3. Scores with three custom evaluators:
       - answer_relevance : does the response address the question?
       - faithfulness     : is the response grounded in retrieved context?
       - conciseness      : is the response reasonably short?
  4. Results appear in the LangSmith dashboard under the project name.

Requires: LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, GROQ_API_KEY
"""

import os
from langsmith import Client, evaluate

from telecom_rag.chain.rag_pipeline import build_chain
from telecom_rag.evaluation.dataset import build_eval_examples, push_to_langsmith
from telecom_rag.config import LANGSMITH_PROJECT

from dotenv import load_dotenv
load_dotenv()


# ── Custom evaluator functions ───────────────────────────────────────────────
# Each takes (run, example) and returns {"key": ..., "score": ...}

def answer_relevance(run, example) -> dict:
    """Heuristic: does the response contain at least one keyword from the
    expected answer?  (Replace with an LLM-as-judge for production.)"""
    prediction = run.outputs.get("output", "")
    reference  = example.outputs.get("answer", "")

    ref_keywords = {w.lower() for w in reference.split() if len(w) > 4}
    pred_words   = {w.lower() for w in prediction.split()}

    if not ref_keywords:
        return {"key": "answer_relevance", "score": 1.0}

    overlap = len(ref_keywords & pred_words) / len(ref_keywords)
    return {"key": "answer_relevance", "score": round(overlap, 3)}


def faithfulness(run, example) -> dict:
    """Heuristic: penalise responses that mention dollar amounts not in the
    expected output (a proxy for hallucinated figures)."""
    import re
    prediction = run.outputs.get("output", "")
    reference  = example.outputs.get("answer", "")

    pred_amounts = set(re.findall(r"\$\d+", prediction))
    ref_amounts  = set(re.findall(r"\$\d+", reference))

    if not pred_amounts:
        return {"key": "faithfulness", "score": 1.0}

    hallucinated = pred_amounts - ref_amounts
    score = 1.0 - (len(hallucinated) / len(pred_amounts))
    return {"key": "faithfulness", "score": round(max(score, 0.0), 3)}


def conciseness(run, example) -> dict:
    """Score 1.0 if under 500 words, linearly decaying to 0 at 1500 words."""
    prediction = run.outputs.get("output", "")
    word_count = len(prediction.split())

    if word_count <= 500:
        score = 1.0
    elif word_count >= 1500:
        score = 0.0
    else:
        score = 1.0 - (word_count - 500) / 1000.0

    return {"key": "conciseness", "score": round(score, 3)}


# ── Runner ───────────────────────────────────────────────────────────────────

def _predict(inputs: dict) -> dict:
    """Wrapper for evaluate(): takes an inputs dict, returns outputs dict."""
    chain = build_chain()
    result = chain.invoke(inputs["question"])
    return {"output": result}


def run(dataset_name: str = "telecom-care-eval") -> None:
    client = Client()

    # Ensure dataset exists
    try:
        client.read_dataset(dataset_name=dataset_name)
    except Exception:
        print("Dataset not found in LangSmith — creating it now …")
        push_to_langsmith(dataset_name)

    print(f"Running evaluation against '{dataset_name}' …\n")
    results = evaluate(
        _predict,
        data=dataset_name,
        evaluators=[answer_relevance, faithfulness, conciseness],
        experiment_prefix="telecom-rag",
        metadata={"project": LANGSMITH_PROJECT},
    )
    print("\nEvaluation complete. View results in the LangSmith dashboard.")
    return results


if __name__ == "__main__":
    run()
