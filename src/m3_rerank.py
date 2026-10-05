from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or not query.strip():
            return []

        model = self._load_model()

        pairs = []
        extracted_docs = []
        for doc in documents:
            text = doc["text"] if isinstance(doc, dict) else getattr(doc, "text", "")
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
            orig_score = float(doc.get("score", 0.0)) if isinstance(doc, dict) else float(getattr(doc, "score", 0.0))
            pairs.append((query, text))
            extracted_docs.append({"text": text, "metadata": meta, "score": orig_score})

        scores = model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif isinstance(scores, (int, float)):
            scores = [float(scores)]
        else:
            scores = [float(s) for s in scores]

        scored = sorted(zip(scores, extracted_docs), key=lambda x: x[0], reverse=True)

        results = []
        for i, (score, doc) in enumerate(scored[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc["score"],
                rerank_score=float(score),
                metadata=doc["metadata"],
                rank=i
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or not query.strip():
            return []
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()

            passages = []
            for i, doc in enumerate(documents):
                text = doc["text"] if isinstance(doc, dict) else getattr(doc, "text", "")
                meta = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
                orig_score = float(doc.get("score", 0.0)) if isinstance(doc, dict) else float(getattr(doc, "score", 0.0))
                passages.append({"id": i, "text": text, "metadata": meta, "score": orig_score})

            rerank_req = RerankRequest(query=query, passages=passages)
            results = self._model.rerank(rerank_req)

            return [
                RerankResult(
                    text=r.get("text", ""),
                    original_score=float(r.get("score", 0.0)),
                    rerank_score=float(r.get("score", 0.0)),
                    metadata=r.get("metadata", {}),
                    rank=idx
                )
                for idx, r in enumerate(results[:top_k])
            ]
        except Exception:
            sorted_docs = sorted(
                documents,
                key=lambda d: d.get("score", 0.0) if isinstance(d, dict) else getattr(d, "score", 0.0),
                reverse=True
            )[:top_k]
            return [
                RerankResult(
                    text=d["text"] if isinstance(d, dict) else getattr(d, "text", ""),
                    original_score=float(d.get("score", 0.0)) if isinstance(d, dict) else float(getattr(d, "score", 0.0)),
                    rerank_score=float(d.get("score", 0.0)) if isinstance(d, dict) else float(getattr(d, "score", 0.0)),
                    metadata=d.get("metadata", {}) if isinstance(d, dict) else getattr(d, "metadata", {}),
                    rank=i
                )
                for i, d in enumerate(sorted_docs)
            ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")