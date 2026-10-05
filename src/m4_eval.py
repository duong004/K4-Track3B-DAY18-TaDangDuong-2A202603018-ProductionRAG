from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_float(val: any) -> float:
    """Helper converting metric values safely to non-NaN float."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (ValueError, TypeError):
        return 0.0


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        df = result.to_pandas()

        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]) if isinstance(row["contexts"], (list, tuple)) else [str(row["contexts"])],
                ground_truth=str(row["ground_truth"]),
                faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_float(row.get("context_precision", 0.0)),
                context_recall=_safe_float(row.get("context_recall", 0.0)),
            ))

        faith_mean = _safe_float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0
        relevancy_mean = _safe_float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else 0.0
        precision_mean = _safe_float(df["context_precision"].mean()) if "context_precision" in df else 0.0
        recall_mean = _safe_float(df["context_recall"].mean()) if "context_recall" in df else 0.0

        return {
            "faithfulness": round(faith_mean, 4),
            "answer_relevancy": round(relevancy_mean, 4),
            "context_precision": round(precision_mean, 4),
            "context_recall": round(recall_mean, 4),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results or bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored_items = []
    for item in eval_results:
        metrics = {
            "faithfulness": _safe_float(item.faithfulness),
            "answer_relevancy": _safe_float(item.answer_relevancy),
            "context_precision": _safe_float(item.context_precision),
            "context_recall": _safe_float(item.context_recall),
        }
        avg_score = sum(metrics.values()) / 4.0
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        scored_items.append((avg_score, worst_metric, worst_score, item))

    scored_items.sort(key=lambda x: x[0])

    failures = []
    for avg_score, worst_metric, worst_score, item in scored_items[:bottom_n]:
        diag, fix = diagnostic_tree.get(worst_metric, ("Unknown issue", "Review pipeline configuration"))
        failures.append({
            "question": item.question,
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "avg_score": round(avg_score, 4),
            "diagnosis": diag,
            "suggested_fix": fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")