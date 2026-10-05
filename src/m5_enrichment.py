from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        api_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
        if api_key and api_key.strip() and not api_key.startswith("sk-placeholder") and len(api_key) > 15:
            try:
                from openai import OpenAI
                _openai_client = OpenAI(api_key=api_key)
            except Exception:
                _openai_client = None
    return _openai_client


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""

    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    if sentences:
        fallback = ". ".join(sentences[:2])
        return fallback if fallback.endswith(".") else f"{fallback}."
    return text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if not text.strip():
        return []

    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content or ""
            questions = [q.strip().lstrip("0123456789.-) ") for q in raw.split("\n") if q.strip()]
            valid_qs = [q for q in questions if len(q) > 5]
            if valid_qs:
                return valid_qs[:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"Thông tin về {s.rstrip('.')} như thế nào?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text.strip():
        return ""

    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
                temperature=0.3,
            )
            context = resp.choices[0].message.content
            if context and context.strip():
                return f"{context.strip()}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    prefix = f"Trích từ tài liệu: {document_title}." if document_title else "Trích từ tài liệu nội bộ."
    return f"{prefix}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if not text.strip():
        return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}

    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về đúng 1 JSON object dạng: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content or "{}"
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if not text.strip():
        return {}

    client = _get_openai_client()
    if client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Phân tích đoạn văn và trả về đúng 1 JSON object có cấu trúc:\n"
                            "{\n"
                            '  "summary": "tóm tắt 2-3 câu",\n'
                            '  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],\n'
                            '  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu và nội dung chính",\n'
                            '  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}\n'
                            "}"
                        ),
                    },
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw.strip())
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trích từ tài liệu: {source}." if source else "Trích từ tài liệu nội bộ.",
        "metadata": {"topic": "general", "entities": [], "category": "policy", "language": "vi"},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.
    Có tích hợp caching tự động để tiết kiệm chi phí và thời gian gọi API.
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods
    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "enrichment_cache.json")

    # Nạp từ cache nếu tồn tại
    if use_combined and os.path.exists(cache_path) and len(chunks) > 5:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if len(cached_data) == len(chunks):
                print(f"  ⚡ Đã nạp thành công {len(cached_data)} chunks từ cache ({cache_path})!", flush=True)
                return [
                    EnrichedChunk(
                        original_text=item["original_text"],
                        enriched_text=item["enriched_text"],
                        summary=item.get("summary", ""),
                        hypothesis_questions=item.get("hypothesis_questions", []),
                        auto_metadata=item.get("auto_metadata", {}),
                        method=item.get("method", "combined"),
                    )
                    for item in cached_data
                ]
        except Exception as e:
            print(f"  ⚠️  Không thể đọc cache: {e}, tiến hành chạy API...", flush=True)

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    # Lưu kết quả vào cache để sử dụng lại
    if use_combined and len(enriched) > 5:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            serializable = [
                {
                    "original_text": c.original_text,
                    "enriched_text": c.enriched_text,
                    "summary": c.summary,
                    "hypothesis_questions": c.hypothesis_questions,
                    "auto_metadata": c.auto_metadata,
                    "method": c.method,
                }
                for c in enriched
            ]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")