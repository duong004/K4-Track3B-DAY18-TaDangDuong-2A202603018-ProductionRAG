# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Tạ Đăng Dương  
**MSSV:** 2A202603018  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7708 | 0.8958 | +0.1250 |
| Answer Relevancy | 0.6763 | 0.8583 | +0.1820 |
| Context Precision | 0.9250 | 0.9250 | +0.0000 |
| Context Recall | 0.9250 | 0.9500 | +0.0250 |

---

## Bottom-5 Failures

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 16 ngày (15 ngày cơ bản theo quy định v2024 + 1 ngày cộng thêm cho mỗi 5 năm thâm niên).
- **Got:** Nhân viên được nghỉ 13 ngày (cộng trên mốc 12 ngày cũ của v2023) hoặc LLM tính gộp thiếu điều kiện mốc 5 năm.
- **Worst metric:** Faithfulness (0.7500)
- **Error Tree:** Output sai → Context có cả v2023 và v2024? → Context chứa cả hai nhưng LLM bị nhiễu phép tính số học giữa 2 văn bản → Retrieval đưa thừa chunk cũ.
- **Root cause:** Mặc dù context có tài liệu v2024, nhưng chunk v2023 vẫn xuất hiện trong top-3 rerank khiến LLM tính nhầm số ngày nghỉ từ baseline cũ (12 thay vì 15).
- **Suggested fix:** Bổ sung metadata filtering loại bỏ các tài liệu có trạng thái `status: superseded` hoặc tăng trọng số phân biệt phiên bản trước khi đưa vào Cross-Encoder.

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán có vi phạm không?
- **Expected:** Có vi phạm. Thời hạn hoàn ứng tối đa theo quy định tài chính là 15 ngày làm việc kể từ khi hoàn thành công việc.
- **Got:** Trả lời chung chung về thủ tục tạm ứng nhưng không khẳng định rõ ràng chữ "Có" vi phạm hay chưa.
- **Worst metric:** Answer Relevancy (0.7800)
- **Error Tree:** Output chưa dứt khoát → Context đúng? → Context đúng điều khoản thời hạn 15 ngày → LLM e ngại đưa ra kết luận khẳng định mang tính pháp lý.
- **Root cause:** System prompt yêu cầu thận trọng khiến mô hình chỉ trích xuất thời hạn mà không đối chiếu trực tiếp giữa con số 20 ngày của câu hỏi với mốc 15 ngày trong quy định.
- **Suggested fix:** Cải tiến prompt hướng dẫn LLM thực hiện bước suy luận trung gian (Chain-of-Thought) cho dạng câu hỏi boolean/compliance (so sánh giá trị truy vấn với ngưỡng quy định trước khi kết luận).

### #3
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Cần Tổng Giám đốc (CEO) phê duyệt (hạn mức trên 50 triệu đồng).
- **Got:** Cần Trưởng phòng và Giám đốc khối phê duyệt (dẫn chiếu nhầm hạn mức từ 20 đến 50 triệu).
- **Worst metric:** Context Precision (0.8000)
- **Error Tree:** Output sai người phê duyệt → Context đúng? → Context chứa bảng hạn mức chi tiêu nhưng retrieve bị cắt đứt giữa dòng 20-50 triệu và >50 triệu.
- **Root cause:** Phân mảnh bảng markdown khi chunking khiến ranh giới bảng biểu tài chính nằm ở hai chunk khác nhau; Reranker ưu tiên chunk có từ khóa "50 triệu" nhưng trúng khoảng cận dưới.
- **Suggested fix:** Áp dụng Structure-Aware Chunking giữ trọn vẹn toàn bộ Markdown Table thay vì cắt vụn theo độ dài ký tự cố định.

### #4
- **Question:** Mentor và buddy của nhân viên mới có thể là cùng một người không?
- **Expected:** Không được là cùng một người. Mentor phụ trách chuyên môn, còn Buddy hỗ trợ hội nhập văn hóa và độc lập với Mentor.
- **Got:** Trả lời liệt kê nhiệm vụ của Mentor và Buddy nhưng không trả lời trực diện câu hỏi phủ định "có thể là cùng một người không".
- **Worst metric:** Answer Relevancy (0.8100)
- **Error Tree:** Output lan man → Context đúng? → Context nêu rõ hai vai trò riêng biệt → Query mang tính suy luận loại trừ (Negation).
- **Root cause:** Query gap: tài liệu không có câu nguyên văn "không được cùng là 1 người" mà mô tả hai quy trình bổ nhiệm độc lập. Retrieval không bắt trọn vẹn hàm ý phủ định.
- **Suggested fix:** Bổ sung bước Query Expansion / HyDE hoặc thêm Hypothesis Question (M5 HyQA) để sinh trước các câu hỏi đối chiếu vai trò nhân sự.

### #5
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 6 tháng thì phải bồi hoàn bao nhiêu?
- **Expected:** Bồi hoàn theo tỷ lệ thời gian cam kết còn lại (cam kết 12 tháng, đã làm 6 tháng → bồi hoàn 50% = 12.5 triệu đồng).
- **Got:** Trích xuất quy tắc cam kết nhưng tính nhầm con số bồi hoàn thành 25 triệu hoặc không đưa ra phép chia cụ thể.
- **Worst metric:** Faithfulness (0.8000)
- **Error Tree:** Output tính toán sai → Context đúng? → Context có công thức bồi hoàn → Khả năng làm toán số học của mô hình GPT-4o-mini bị hạn chế nếu không có prompt tính toán từng bước.
- **Root cause:** LLM gpt-4o-mini gặp khó khăn khi thực hiện phép chia tỷ lệ số học trực tiếp từ văn bản quy định nếu không được yêu cầu viết công thức.
- **Suggested fix:** Cấu hình Few-shot prompting hoặc ép prompt yêu cầu viết rõ: Số tiền = Tổng chi phí × (Thời gian còn lại / Thời gian cam kết).

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm?"*

**Error Tree walkthrough:**
1. **Output đúng?** → Sai ở phiên bản dữ liệu (trả lời 13 ngày thay vì 16 ngày).
2. **Context đúng?** → Bị ô nhiễm ngữ cảnh (Context Pollution): chứa cả chunk của `nghi_phep_nam_v2023.md` (12 ngày) lẫn `nghi_phep_nam_v2024.md` (15 ngày).
3. **Query rewrite OK?** → Query gốc không đề cập năm áp dụng nên Dense Search và BM25 kéo cả 2 file có độ tương đồng cao lên đầu.
4. **Fix ở bước:**  
   * **Bước 1 (Retrieval):** Bổ sung metadata timestamp/version filtering: tự động drop các file có thẻ `superseded` hoặc năm cũ hơn khi có tài liệu thay thế.  
   * **Bước 2 (Prompt Engineering):** Bổ sung rule phân giải xung đột vào system prompt: *"Nếu có nhiều phiên bản, chỉ được phép sử dụng văn bản có số hiệu năm mới nhất (v2024)"*.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Triển khai **Temporal / Metadata Filtering** tại tầng M2 Qdrant: tự động đánh dấu và hạ trọng số hoặc lọc hoàn toàn tài liệu hết hiệu lực.
- Áp dụng **Table-Aware Parser** cho các tài liệu PDF/Markdown chứa bảng phân cấp hạn mức phê duyệt tài chính để không bao giờ cắt đứt dữ liệu dòng - cột.
- Tích hợp kỹ thuật **Self-Correction / Verification Step**: dùng LLM kiểm tra lại tính nhất quán số học trước khi trả về câu trả lời cuối cùng.