from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks or []
        self.corpus_tokens = []
        if not self.documents:
            self.bm25 = None
            return

        for chunk in self.documents:
            text = chunk["text"] if isinstance(chunk, dict) else chunk.text
            tokens = segment_vietnamese(text).lower().split()
            self.corpus_tokens.append(tokens)

        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        indexed_scores = [(i, float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed_scores[:top_k]

        results = []
        for idx, score in top_indices:
            doc = self.documents[idx]
            text = doc["text"] if isinstance(doc, dict) else doc.text
            meta = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
            results.append(SearchResult(text=text, score=score, metadata=meta, method="bm25"))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            return

        from qdrant_client.models import Distance, VectorParams, PointStruct

        texts = [c["text"] if isinstance(c, dict) else c.text for c in chunks]
        encoder = self._get_encoder()
        vectors = encoder.encode(texts, show_progress_bar=False)
        dim = vectors.shape[1] if hasattr(vectors, "shape") else len(vectors[0])

        try:
            if hasattr(self.client, "collection_exists") and self.client.collection_exists(collection):
                self.client.delete_collection(collection)
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        except Exception:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

        points = []
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            text = c["text"] if isinstance(c, dict) else c.text
            meta = dict(c.get("metadata", {})) if isinstance(c, dict) else dict(getattr(c, "metadata", {}))
            payload = {**meta, "text": text}
            vec_list = v.tolist() if hasattr(v, "tolist") else list(v)
            points.append(PointStruct(id=i, vector=vec_list, payload=payload))

        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query.strip():
            return []

        try:
            query_vector = self._get_encoder().encode(query).tolist()
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
                points = response.points
            else:
                points = self.client.search(collection_name=collection, query_vector=query_vector, limit=top_k)

            return [
                SearchResult(
                    text=pt.payload.get("text", ""),
                    score=float(pt.score),
                    metadata=pt.payload,
                    method="dense"
                )
                for pt in points
            ]
        except Exception:
            return []


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1)."""
    if not results_list:
        return []

    rrf_scores: dict[str, dict] = {}
    for r_list in results_list:
        for rank, result in enumerate(r_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "result": result
                }
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    sorted_entries = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)[:top_k]

    return [
        SearchResult(
            text=entry["result"].text,
            score=float(entry["score"]),
            metadata=entry["result"].metadata,
            method="hybrid"
        )
        for entry in sorted_entries
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")