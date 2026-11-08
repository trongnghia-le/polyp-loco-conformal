# 2026-11-08 — Kiểm tra khả thi hướng B: "lời bảo đảm conformal trên nội soi bị thổi phồng vì chia theo khung hình"

## Đã kiểm chứng
1. **CVC-ClinicDB có bảng ghép khung → video chính thức** (trang https://polyp.grand-challenge.org/CVCClinicDB/, ảnh Table 1 `cvc-clinicdbgt2.png`, đã xem): **29 chuỗi**, khung đánh số liên tục theo chuỗi:
   1: 1–25 · 2: 26–50 · 3: 51–67 · 4: 68–78 · 5: 79–103 · 6: 104–126 · 7: 127–151 · 8: 152–177 · 9: 178–199 · 10: 200–205 · 11: 206–227 · 12: 228–252 · 13: 253–277 · 14: 278–297 · 15: 298–317 · 16: 318–342 · 17: 343–363 · 18: 364–383 · 19: 384–408 · 20: 409–428 · 21: 429–447 · 22: 448–466 · 23: 467–478 · 24: 479–503 · 25: 504–528 · 26: 529–546 · 27: 547–571 · 28: 572–591 · 29: 592–612.
   (Bài benchmark 2017 ghi "31 sequences, 23 patients" — mâu thuẫn với bảng chính thức 29 chuỗi; dùng bảng chính thức, ghi chú mâu thuẫn.) Giấy phép: chỉ nghiên cứu/giáo dục, cấm thương mại; trích Bernal et al. 2015 (CMIG 43:99–111). **Trang chính thức không còn link tải** (đoạn "following link" rỗng) → cần lấy bản sao giữ nguyên số khung (ví dụ bản dùng trong PraNet hoặc bản trên Kaggle) và kiểm tra số khung.
2. **Ví dụ polyp trong bài Conformal Risk Control (Angelopoulos et al.) chia theo từng ảnh:** `polyps/polyp_utils.py` gộp `['HyperKvasir','CVC-300','CVC-ClinicDB','Kvasir','CVC-ColonDB','ETIS-LaribPolypDB']`; `polyps/risk_histogram.py` dùng `torch.randperm` trên bảng loss theo ảnh rồi lấy `num_calib = 1000` ảnh đầu làm hiệu chuẩn (github.com/aangelopoulos/conformal-risk, tải ngày 2026-11-08).
3. **Đã có bằng chứng rò rỉ do chia ngẫu nhiên theo khung ở ClinicDB cho phân vùng** (FCB-SwinV2, arXiv 2302.01027: chia không rò rỉ làm điểm giảm mạnh) — tức là ý "khung cùng video làm điểm bị thổi phồng" đã được biết cho Dice/IoU, **chưa thấy cho lời bảo đảm conformal** (trong tìm kiếm đến 2026-11-08).

## Điểm tinh tế (phải viết đúng)
Chia ngẫu nhiên theo khung làm hiệu chuẩn và kiểm tra **hoán đổi được theo đúng định nghĩa**, nên lời bảo đảm vẫn đúng — nhưng cho **"một khung mới từ các video đã có"**, không phải **"một bệnh nhân/video mới"**. Trên PolypGen (trong miền, 3 seed): chia theo ảnh lệch tối đa +0,0034 (đúng), chia theo cụm video +0,0427 (vượt). Tuyên bố đúng là: *lời bảo đảm được chứng nhận cho sai tổng thể*; mức chênh trong miền ~4 điểm %, khi đổi bệnh viện thì lớn hơn nhiều.

## Kế hoạch đề xuất
- **B1 (CPU, không train):** dùng 18 mô hình YOLO11s (train trên PolypGen, chưa thấy ClinicDB) suy luận 612 khung ClinicDB → (i) CRC trong ClinicDB chia theo khung vs theo video; (ii) kiểm định ngoài: hiệu chuẩn trên PolypGen, đo trên ClinicDB (một bệnh viện thứ 7).
- **B2 (Kaggle, ~2–4 giờ GPU):** tái hiện giao thức phổ biến (train trên Kvasir + ClinicDB chia ngẫu nhiên theo kiểu PraNet) vs chia theo video → đo AP và lời bảo đảm bị thổi phồng bao nhiêu khi mô hình đã thấy video đó lúc huấn luyện.
- Rủi ro: kết quả có thể chỉ chênh vài điểm % → khi đó bài vẫn có kiểm định ngoài trên bộ dữ liệu thứ hai.

## Dữ liệu đã lấy (2026-11-08)
- Bản Kaggle `balraj98/cvcclinicdb` (tải bằng Kaggle CLI) → `data/raw/cvc-clinicdb/` (không đưa lên git): 612 ảnh PNG + 612 mask (cả bản TIF), số khung 1–612, ảnh 384×288.
- `metadata.csv` của bản này có `sequence_id`: **khớp 100 % bảng chính thức** (29 chuỗi, 6–26 khung/chuỗi, trung vị 22).
- Kiểm tra độc lập: độ khác trung bình giữa hai khung liền kề (ảnh xám 96×72) — trong chuỗi trung vị 13,9; tại ranh giới chuỗi chính thức trung vị 39,1 (nhỏ nhất 19,5); không ranh giới nào trông liền mạch.
- B1 giao cho agent (`scripts/predict_clinicdb.py`, `scripts/clinicdb_crc.py`, `scripts/fig_clinicdb.py`, ghi chú `2026-11-08_clinicdb-b1.md`).
