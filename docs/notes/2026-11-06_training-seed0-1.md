# 2026-11-06 — Huấn luyện YOLO11s: seed 0 và seed 1 xong (12/12 lần chạy đạt)

## Đã làm
- Notebook `run_seed0` (version 2) và `run_seed1` chạy song song trên Kaggle (Tesla T4, 1 GPU/lần chạy), đẩy bằng Kaggle CLI 2.2.5 (`scripts/push_when_free.sh`, `scripts/watch_kaggle.sh`). Mỗi notebook ~4,8 giờ.
- Seed 2 tự được đẩy lúc 17:06 khi có chỗ GPU trống (Kaggle cho tối đa 2 phiên chạy nền cùng lúc).
- Kết quả tải về `outputs/kaggle/<fold>_seed<k>/` (preds parquet, run_meta, best.pt, index, results.csv, log); tổng hợp `outputs/kaggle/summary_all.csv`.

## Sự cố trước đó (đã sửa)
- Version 1 của seed 0: C1 và C2 huấn luyện xong nhưng hết bộ nhớ GPU khi dự đoán tập cal. Nguyên nhân: Ultralytics nạp **một danh sách đường dẫn thành một lô duy nhất** (`LoadPilAndNumpy`, `bs = len(list)`, bỏ qua `batch`); cỡ phân bổ 4,50/3,99 GiB khớp tỷ lệ số ảnh cal 1474/1308. Sửa: đưa ảnh theo từng khúc 16 ảnh (`--pred-batch`). Kiểm chứng bằng notebook `run_test` (C2, 1 epoch, 8,2 phút) trước khi chạy thật.
- Chẩn đoán đầu tiên (bộ nhớ do trainer giữ lại) là sai.

## Số liệu (từ `run_meta_*.json`; mAP trên tập val = trung tâm nguồn, chưa phải trung tâm bị loại)
| fold | seed | epochs | best | train (phút) | val mAP50 | val mAP50-95 | số box |
|---|---|---|---|---|---|---|---|
| C1 | 0 | 50 | 45 | 59,1 | 0,732 | 0,583 | 24.181 |
| C1 | 1 | 29 | 19 | 46,1 | 0,672 | 0,521 | 51.309 |
| C2 | 0 | 49 | 41 | 50,4 | 0,685 | 0,505 | 25.029 |
| C2 | 1 | 50 | 42 | 48,7 | 0,696 | 0,509 | 17.270 |
| C3 | 0 | 29 | 19 | 34,8 | 0,676 | 0,524 | 39.017 |
| C3 | 1 | 41 | 31 | 46,4 | 0,728 | 0,551 | 24.495 |
| C4 | 0 | 50 | 49 | 47,9 | 0,797 | 0,658 | 18.342 |
| C4 | 1 | 45 | 35 | 40,3 | 0,784 | 0,647 | 48.246 |
| C5 | 0 | 41 | 31 | 47,1 | 0,754 | 0,586 | 20.034 |
| C5 | 1 | 44 | 34 | 47,4 | 0,746 | 0,572 | 19.495 |
| C6 | 0 | 42 | 32 | 38,2 | 0,706 | 0,544 | 15.324 |
| C6 | 1 | 50 | 40 | 43,2 | 0,738 | 0,562 | 33.849 |

## Nhận xét / còn mở
- Dừng sớm có tác dụng ở 8/12 lần (29–49 epoch). Biến thiên giữa seed khá lớn ở C1 (mAP50 0,732 vs 0,672) và C3 → cần 3 seed và báo độ dao động.
- Số box dự đoán (ngưỡng 0,001) dao động 15k–51k/lần chạy; bình thường cho conformal (cần đủ box điểm thấp).
- Chưa có số trên trung tâm bị loại (target) — sẽ tính ở bước đánh giá/conformal.
- Mỗi lần chạy giữ lại các ảnh có 0 box? Ảnh không có box nào không có dòng trong parquet; dùng `index_<fold>.csv` làm danh sách đầy đủ.
- Hình: chưa vẽ (bảng huấn luyện không phải hình cho bài); hình đầu tiên có giá trị sẽ là kết quả trên trung tâm bị loại.
