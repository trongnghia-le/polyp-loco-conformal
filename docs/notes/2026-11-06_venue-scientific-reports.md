# 2026-11-06 — Yêu cầu của Scientific Reports

Nguồn: bản lưu trang "Submission guidelines | Scientific Reports" ngày 20/03/2026 (trang gốc nature.com/srep/author-instructions/submission-guidelines bắt đăng nhập; cần đọc lại bản sống trước khi nộp).

## Hình thức bài (Article)
- Lý tưởng ≤ 11 trang dàn trang; **thân bài ≤ 4.500 từ** (không tính Abstract, Methods, References, chú thích hình).
- **Tiêu đề ≤ 20 từ**, một câu khoa học chính xác, không chơi chữ.
- **Abstract ≤ 200 từ**, không trích dẫn, không chia mục, vừa là giới thiệu chung vừa tóm tắt kết quả cho người ngoài ngành. Không có graphical abstract.
- Tối đa 6 từ khoá.
- Cấu trúc gợi ý: Introduction → Results (có tiêu đề con) → Discussion (không tiêu đề con) → **Methods (không giới hạn từ; phải đủ để tái lập)**.
- Sau đó: References (**≤ 60**, không ép chặt; kiểu Nature, đánh số trong ngoặc vuông; không nhận .bib, nộp .bbl), Acknowledgements, **Author contributions**, **Data availability (bắt buộc)**, Competing interests (bắt buộc, cho từng tác giả), chú thích hình (**≤ 350 từ/hình**), bảng (≤ 1 trang). Không dùng chú thích chân trang.
- **Tối đa 8 display items (hình + bảng).** Phụ lục/Supplementary nộp riêng một file PDF; không đặt phụ lục trong thân bài.
- Lần nộp đầu: Word/LaTeX/PDF một file ≤ 3 MB được; có template Overleaf (`wlscirep.cls`). Bản sửa: Word hoặc LaTeX, một cột, đánh số trang.
- Cover letter: thông tin tác giả liên hệ, vì sao hợp Sci Rep, gợi ý/loại trừ reviewer, đã trao đổi với Editorial Board Member chưa.

## Chính sách cần chú ý
- **LLM (ChatGPT, Claude…) không được đứng tên tác giả; việc dùng LLM phải được mô tả trong Methods.** → Repo không ghi Claude là contributor là đúng, nhưng bài báo vẫn phải khai báo việc dùng LLM theo chính sách này.
- Dữ liệu người: PolypGen là dữ liệu công khai đã ẩn danh — cần một câu đạo đức dẫn về phê duyệt/đồng thuận của bộ dữ liệu gốc (Sci Data 2023) và giấy phép CC BY 4.0. Không để tên/ID bệnh nhân trong hình (một số khung có lớp chữ "ID patient/Nom patient" — kiểm tra trước khi đưa ảnh nội soi vào hình).
- Ảnh lấy từ nơi khác phải có quyền; nêu nguồn ảnh và phần mềm (tên, phiên bản, URL).

## Hình (đã đưa vào `configs/paper.mplstyle` và CLAUDE.md)
- Chữ sans-serif (Helvetica/Arial), cùng font và cỡ chữ ở mọi hình; nền trắng; không trang trí thừa, không 3D; không cắt trục histogram.
- **Nét mảnh nhất ≥ 1 pt** ở kích thước cuối (style cũ dùng 0,6 pt → đã sửa thành 1,0).
- Biểu đồ nộp dạng vector (EPS/AI; xuất thêm `.eps` cạnh `.pdf`); ảnh chụp dạng TIFF/JPG; hình nhiều ô ghép thành một file (a, b, c).
- Thanh sai số khi phù hợp; chú thích phải nói rõ ± là s.d. hay s.e.m.
- Chưa tìm thấy độ rộng cột (mm) trong bản lưu — vẫn dùng 89/183 mm (chuẩn Nature) cho tới khi kiểm bản sống.

## Thống kê
- Nêu tên kiểm định, n, so sánh quan tâm, lý do chọn kiểm định, α, một/hai phía, **giá trị P thực** (không chỉ "P < 0,05"); "significant" phải đi kèm P; điều chỉnh so sánh bội. Báo n, trung tâm (mean/median) và độ phân tán (SD/khoảng); dữ liệu nhỏ dùng khoảng (range).
- Hợp với kế hoạch: bootstrap theo cụm video, CI 95% theo từng trung tâm; Wilcoxon theo trung tâm (6 trung tâm) không thể cho P < ~0,031 → báo CI là chính.

## Ảnh hưởng tới kế hoạch
- 8 display items: dự kiến (1) thành phần dữ liệu + quy trình LOCO, (2) sơ đồ phương pháp, (3) baseline theo trung tâm, (4) tỷ lệ bỏ sót thực tế theo α × trung tâm, (5) ngân sách nhãn đích (20/50/100) vs tần suất vi phạm, (6) cảnh báo giả, (7) bảng tổng hợp, (8) dự phòng. SAM2 (nếu giữ) đưa vào Supplementary.
- 4.500 từ thân bài + Methods không giới hạn → để chi tiết kỹ thuật (quy tắc đếm polyp, phát hiện lỗi gán trung tâm của bài gốc, cụm video) trong Methods/Supplementary.
