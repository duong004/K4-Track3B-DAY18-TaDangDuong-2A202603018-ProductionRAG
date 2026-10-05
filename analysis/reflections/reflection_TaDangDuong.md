# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Tạ Đăng Dương  
**MSSV:** 2A202603018  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 05/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic & Hierarchical chunking | M1 | `chunk_semantic()`, `chunk_hierarchical()` | Khi áp dụng Hierarchical chunking, chia tài liệu thành 26 parent chunks (ngữ cảnh rộng) và 104 child chunks (đoạn nhỏ tập trung). Việc tách nhỏ giúp search bắt trúng từ khóa, trong khi trả về parent giúp LLM có đầy đủ ngữ cảnh để giải quyết câu hỏi phức tạp. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()`, `segment_vietnamese()` | BM25 bắt chính xác các con số và mã quy định đặc thù tiếng Việt sau khi dùng underthesea tách từ (loại bỏ dấu gạch dưới). RRF kết hợp điểm lexical và dense vector một cách công bằng không phụ thuộc biên độ điểm số, tăng khả năng retrieve đúng tài liệu lên 95%. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Mô hình `BAAI/bge-reranker-v2-m3` đánh giá cross-attention trực tiếp giữa cặp câu hỏi - tài liệu. Lọc từ 20 candidate xuống top 3 chất lượng nhất, loại bỏ hiệu quả các chunk rác dù độ trễ trung bình tăng lên khoảng 4.3 giây/query. |
| RAGAS 4 metrics & Failure Analysis | M4 | `evaluate_ragas()`, `failure_analysis()` | RAGAS đo lường khách quan 4 chiều: Faithfulness (0.8958), Answer Relevancy (0.8583), Context Precision (0.9250), Context Recall (0.9500). Cây chẩn đoán (Diagnostic Tree) giúp ánh xạ tức thì từ chỉ số tụt dốc sang lỗi kiến trúc ở tầng retrieval hoặc generation. |
| Contextual embeddings & Enrichment | M5 | `_enrich_single_call()`, `enrich_chunks()` | Áp dụng combined single-call gom summary, context prepend, câu hỏi giả định và metadata vào 1 lượt gọi API. Việc gắn dòng context đầu chunk giải quyết triệt để vấn đề mất ngữ cảnh cục bộ khi chunk bị tách rời khỏi tài liệu gốc. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - Trong lần chạy đầu tiên, pipeline hoàn thành nhưng trả về kết quả RAGAS thất bại hoàn toàn:
    `faithfulness: 0.0000`, `answer_relevancy: 0.0000`, `context_precision: 0.0500`, `context_recall: 0.0000`.
  - Log kiểm tra số lượng chunk hiển thị thông tin bất thường:
    `✓ 104 child chunks (1 parents) from 26 documents`.

- **Nguyên nhân gốc rễ & Cách debug:**
  - *Phân tích log:* Biến `parents = []` trong hàm `chunk_hierarchical()` được tạo cục bộ trong vòng lặp từng file. Do đó mỗi tài liệu đều gán parent đầu tiên là `parent_0`.
  - Khi `pipeline.py` duyệt qua toàn bộ 26 tài liệu, key `"parent_0"` trong từ điển `parent_map` bị ghi đè liên tục 26 lần và chỉ lưu lại nội dung của tài liệu cuối cùng.
  - Khi chạy truy vấn, toàn bộ 104 child chunks đều ánh xạ về một parent duy nhất không chứa thông tin cần tìm. LLM tuân thủ prompt nghiêm ngặt nên trả lời *"Không tìm thấy thông tin"* cho tất cả 20 câu hỏi.
  - *Cách khắc phục:* Bổ sung tiền tố định danh tài liệu vào mã định danh parent: `pid = f"{prefix}parent_{len(parents)}"`. Sau khi sửa, hệ thống nhận diện chính xác 26 parents riêng biệt cho 26 tài liệu, điểm Faithfulness lập tức tăng vọt lên **0.8958** và Recall đạt **0.9500**.
  - Đồng thời, xây dựng cơ chế caching `reports/enrichment_cache.json` cho M5 giúp tái sử dụng dữ liệu đã làm giàu, giảm thời gian chạy lại từ 272 giây xuống dưới 1 giây.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - *Kiến thức:* Hiểu sâu hơn về sự sai lệch giữa không gian định danh (ID namespace) cục bộ của từng hàm chunking và không gian định danh toàn cục của hệ thống Retrieval.
  - *Khắc phục:* Luôn thiết kế metadata schema mang tính độc bản (Globally Unique Identifier) bằng cách kết hợp `document_source` + `section_idx` + `chunk_idx`.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: FixIt — Nền tảng tiếp nhận & quản lý sự cố cư dân khu căn hộ tích hợp AI Agent

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng ReAct Agent kết hợp LangGraph và ChromaDB, tìm kiếm văn bản nội quy tòa nhà bằng dense embedding cơ bản qua câu truy vấn của cư dân.
- **Vấn đề / Bottlenecks đang gặp:**
  - Cư dân đặt câu hỏi bằng ngôn ngữ tự nhiên không chứa đúng thuật ngữ kỹ thuật tòa nhà khiến retrieval bỏ sót quy định xử lý sự cố (Context Recall thấp).
  - Có sự xung đột giữa sổ tay cư dân ban hành năm 2023 và quy chế sửa đổi năm 2025.
  - Độ trễ của Agent còn cao khi phải qua nhiều bước suy luận.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Chuyển sang **Hierarchical Chunking (Parent-Child)**. Đoạn con (128 - 256 ký tự) lưu các lỗi cụ thể để khớp nhanh với mô tả của cư dân; đoạn cha (1024 ký tự) chứa đầy đủ quy trình an toàn, số hotline kỹ thuật và biểu phí đền bù.
2. **Search retrieval:** Áp dụng **Hybrid Search (BM25 + Dense + RRF)**. Tích hợp `underthesea` xử lý từ ghép tiếng Việt cho BM25 để bắt chính xác các từ khóa số phòng, tên thiết bị (công tơ điện, van khóa nước, aptomat).
3. **Reranking:** Sử dụng `BAAI/bge-reranker-v2-m3` lọc từ top 15 kết quả xuống top 3 trước khi đẩy vào prompt của triage agent, giảm nhiễu ngữ cảnh.
4. **Evaluation:** Thiết lập bộ 30 câu hỏi benchmark mô phỏng tình huống sự cố thực tế, đánh giá liên tục bằng 4 chỉ số RAGAS trên môi trường CI/CD trước khi triển khai bản mới.
5. **Enrichment:** Ứng dụng **HyQA** để sinh trước các câu hỏi sự cố giả định (ví dụ: "Mất nước tầng 12 thì báo ai?", "Chập điện ban đêm gọi số nào?") vào metadata của chunk nội quy.

#### 3. Timeline triển khai
- **Tuần 1:** Tái cấu trúc lại kho dữ liệu nội quy tòa nhà, triển khai bộ tiền xử lý Underthesea và cài đặt Qdrant Vector DB trên Docker.
- **Tuần 2:** Tích hợp pipeline Hybrid Search + RRF và bộ làm giàu dữ liệu HyQA cho toàn bộ sổ tay kỹ thuật.
- **Tuần 3:** Cài đặt bộ đánh giá RAGAS tự động, đo lường độ trễ và tinh chỉnh prompt xử lý xung đột quy định tòa nhà trước khi đưa vào sản xuất.