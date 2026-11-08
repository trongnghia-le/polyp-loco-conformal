# 2026-11-08 — Chốt phạm vi (scope freeze)

**Quyết định của người dùng:** mục tiêu là một bài Q1 đúng hạn; chấp nhận một nghiên cứu đánh giá chắc chắn nhưng khiêm tốn; **không mở rộng thêm thí nghiệm**. Bài được viết trung thực như một *evaluation study*, không tuyên bố phương pháp mới.

## Trong bài
1. Kiểm tra dữ liệu PolypGen: gán trung tâm cho mọi chuỗi, 3 mâu thuẫn trong bài gốc (seq11, Fig. 2a, seq5 → rủi ro rò rỉ), cụm video, quy tắc đếm polyp.
2. Giao thức LOCO chống rò rỉ (6 vòng, cụm video), YOLO11s × 3 seed (18 lần chạy).
3. Detection: sụt khi đổi bệnh viện; "sàn bỏ sót" (recall tối đa) giới hạn α khả thi.
4. CRC tỷ lệ bỏ sót: gộp nguồn vỡ ở 5/6 trung tâm; ngưỡng an toàn nhất; hiệu chuẩn lại ở trung tâm đích.
5. Đơn vị hoán đổi là video: CRC theo cụm sửa mức vượt trong miền; gán nhãn 1 khung/video.
6. Ablation (Supplementary): YOLO11m seed 0 — không hạ sàn bỏ sót.
7. Kiểm định ngoài trên CVC-ClinicDB (B1, đang chạy lúc chốt — thí nghiệm cuối cùng, không tốn GPU).

## Ngoài bài (không làm)
B2 (train lại trên Kvasir + ClinicDB), CRC có trọng số (strategy d — ghi là hạn chế), YOLO11m seed 1–2 (chỉ chạy nếu còn quota lúc cuối, xem memory), ảnh 960 px, chỉ số bỏ sót theo chuỗi, YOLO→SAM2, bộ dữ liệu thứ ba.

## Việc còn lại để nộp
- Viết lại khung bài theo hướng đánh giá trung thực; điền số 3 seed; Results theo cụm; Discussion; Limitations.
- Hành chính: đồng tác giả viên chức UEH; hỏi tác giả PolypGen về giấy phép (CC BY vs CC BY-NC-SA); khai báo dùng LLM; kiểm JCR 6/2027; định dạng Sci Rep (≤ 4.500 từ, ≤ 8 hình/bảng).
