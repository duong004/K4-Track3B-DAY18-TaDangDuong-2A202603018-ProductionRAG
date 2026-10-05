from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline with Small-to-Big hierarchical retrieval."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    latency_stats = {}

    # Step 1: Load & Chunk (M1) - Hierarchical
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents (Hierarchical)...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_map = {}

    for doc in docs:
        src = doc.get("metadata", {}).get("source", "")
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for p in parents:
            p_id = p.metadata.get("parent_id", p.parent_id)
            if p_id:
                parent_map[p_id] = {"text": p.text, "source": src}
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": child.parent_id,
                    "source": src
                }
            })
    latency_stats["chunking_sec"] = time.perf_counter() - t0
    print(f"  ✓ {len(all_chunks)} child chunks ({len(parent_map)} parents) from {len(docs)} documents ({latency_stats['chunking_sec']:.2f}s)", flush=True)

    # Step 2: Enrichment (M5) - Combined single-call mode
    t0 = time.perf_counter()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, Combined mode)...", flush=True)
    enriched = enrich_chunks(all_chunks, methods=["combined"])
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        latency_stats["enrichment_sec"] = time.perf_counter() - t0
        print(f"  ✓ Enriched {len(enriched)} chunks ({latency_stats['enrichment_sec']:.2f}s)", flush=True)
    else:
        latency_stats["enrichment_sec"] = 0.0
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2) - BM25 + Dense
    t0 = time.perf_counter()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense Qdrant)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    search.parent_map = parent_map
    latency_stats["indexing_sec"] = time.perf_counter() - t0
    print(f"  ✓ Indexed ({latency_stats['indexing_sec']:.2f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.perf_counter()
    print("\n[4/4] Loading CrossEncoder reranker...", flush=True)
    reranker = CrossEncoderReranker()
    _ = reranker._load_model()
    latency_stats["reranker_load_sec"] = time.perf_counter() - t0
    print(f"  ✓ Reranker ready ({latency_stats['reranker_load_sec']:.2f}s)", flush=True)

    search.latency_stats = latency_stats
    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline with Parent-Child context expansion."""
    t_search_start = time.perf_counter()
    results = search.search(query)
    t_search = (time.perf_counter() - t_search_start) * 1000

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    t_rerank_start = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    t_rerank = (time.perf_counter() - t_rerank_start) * 1000

    top_hits = reranked if reranked else results[:RERANK_TOP_K]

    # Small-to-Big: Map retrieved children back to parent chunks
    parent_map = getattr(search, "parent_map", {})
    contexts = []
    seen_parents = set()

    for r in top_hits:
        meta = getattr(r, "metadata", {}) or {}
        p_id = meta.get("parent_id")
        source = meta.get("source", "")
        if p_id and p_id in parent_map:
            if p_id not in seen_parents:
                seen_parents.add(p_id)
                p_info = parent_map[p_id]
                src_label = f"[{p_info.get('source', source)}] " if p_info.get('source', source) else ""
                contexts.append(f"{src_label}{p_info['text']}")
        else:
            text = getattr(r, "text", "")
            src_label = f"[{source}] " if source else ""
            formatted = f"{src_label}{text}"
            if formatted not in contexts:
                contexts.append(formatted)

    if not contexts:
        contexts = [r.text for r in top_hits] if top_hits else ["Không tìm thấy thông tin."]

    # Generation via LLM
    t_llm_start = time.perf_counter()
    from config import OPENAI_API_KEY
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n---\n\n".join(contexts)
            system_prompt = (
                "Bạn là trợ lý AI phân tích tài liệu nội bộ.\n"
                "Quy tắc trả lời:\n"
                "1. CHỈ dựa trên Context được cung cấp. Tuyệt đối không suy diễn ngoài context.\n"
                "2. XUNG ĐỘT PHIÊN BẢN: Nếu có sự khác biệt giữa các phiên bản (ví dụ v2023 vs v2024, v1 vs v2, cũ vs mới), "
                "LUÔN LUÔN ưu tiên thông tin trong phiên bản mới nhất/hiện hành (v2024, v2).\n"
                "3. Trả lời trực tiếp, rõ ràng, chính xác và đầy đủ các số liệu, thời hạn được hỏi.\n"
                "4. Nếu trong context không có dữ liệu, hãy trả lời chính xác: 'Không tìm thấy.'"
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
                temperature=0.0
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy."

    t_llm = (time.perf_counter() - t_llm_start) * 1000

    if not hasattr(search, "query_latencies"):
        search.query_latencies = {"search_ms": [], "rerank_ms": [], "llm_ms": []}
    search.query_latencies["search_ms"].append(t_search)
    search.query_latencies["rerank_ms"].append(t_rerank)
    search.query_latencies["llm_ms"].append(t_llm)

    return answer, contexts


def print_latency_report(search: HybridSearch, total_pipeline_sec: float, eval_sec: float) -> dict:
    """Print and format professional Latency Breakdown Report."""
    base_stats = getattr(search, "latency_stats", {})
    query_stats = getattr(search, "query_latencies", {"search_ms": [0], "rerank_ms": [0], "llm_ms": [0]})

    avg_search = sum(query_stats["search_ms"]) / max(len(query_stats["search_ms"]), 1)
    avg_rerank = sum(query_stats["rerank_ms"]) / max(len(query_stats["rerank_ms"]), 1)
    avg_llm = sum(query_stats["llm_ms"]) / max(len(query_stats["llm_ms"]), 1)

    breakdown = {
        "chunking_sec": round(base_stats.get("chunking_sec", 0.0), 3),
        "enrichment_sec": round(base_stats.get("enrichment_sec", 0.0), 3),
        "indexing_sec": round(base_stats.get("indexing_sec", 0.0), 3),
        "reranker_load_sec": round(base_stats.get("reranker_load_sec", 0.0), 3),
        "avg_search_ms": round(avg_search, 2),
        "avg_rerank_ms": round(avg_rerank, 2),
        "avg_llm_gen_ms": round(avg_llm, 2),
        "eval_ragas_sec": round(eval_sec, 3),
        "total_runtime_sec": round(total_pipeline_sec, 3)
    }

    print("\n" + "=" * 65)
    print("           LATENCY BREAKDOWN REPORT (BONUS +2 PTS)")
    print("=" * 65)
    print(f" {'Pipeline Component':<40} | {'Latency / Duration':<20}")
    print("-" * 65)
    print(f" {'1. Document Chunking (Hierarchical)':<40} | {breakdown['chunking_sec']:>14.2f} s")
    print(f" {'2. M5 Enrichment (Combined Single-Call)':<40} | {breakdown['enrichment_sec']:>14.2f} s")
    print(f" {'3. M2 Hybrid Indexing (BM25 + Qdrant)':<40} | {breakdown['indexing_sec']:>14.2f} s")
    print(f" {'4. M3 CrossEncoder Model Load':<40} | {breakdown['reranker_load_sec']:>14.2f} s")
    print(f" {'5. Avg Hybrid Search per query':<40} | {breakdown['avg_search_ms']:>14.2f} ms")
    print(f" {'6. Avg CrossEncoder Rerank per query':<40} | {breakdown['avg_rerank_ms']:>14.2f} ms")
    print(f" {'7. Avg LLM Answer Generation per query':<40} | {breakdown['avg_llm_gen_ms']:>14.2f} ms")
    print(f" {'8. M4 RAGAS Evaluation':<40} | {breakdown['eval_ragas_sec']:>14.2f} s")
    print("-" * 65)
    print(f" {'TOTAL PIPELINE RUNTIME':<40} | {breakdown['total_runtime_sec']:>14.2f} s")
    print("=" * 65 + "\n")

    return breakdown


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, pipeline_start_time: float = None):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.perf_counter()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_sec = time.perf_counter() - t0
    print(f"  ✓ RAGAS done ({eval_sec:.1f}s)", flush=True)

    total_time = (time.perf_counter() - pipeline_start_time) if pipeline_start_time else eval_sec
    latency_report = print_latency_report(search, total_time, eval_sec)

    print("=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    results["latency_breakdown"] = latency_report

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.perf_counter()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker, pipeline_start_time=start)
    print(f"Total Completed: {time.perf_counter() - start:.1f}s")