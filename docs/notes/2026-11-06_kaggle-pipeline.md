# 2026-11-06 — Pipeline YOLO trên Kaggle (1 fold LOCO)

## Đã làm
- `requirements.txt` (pin chính xác; ultralytics 8.4.170 = bản 2026-09-30, cũ hơn một tháng so với 8.4.202 mới nhất lúc chọn; torch 2.14.1 CPU kéo theo). `.venv` đã tạo, `pip install -r requirements.txt` chạy sạch. Ultralytics kéo thêm `opencv-python` (không headless) cạnh `opencv-python-headless`; cùng phiên bản 5.0.0.93, không xung đột. Ultralytics là AGPL-3.0 (ghi trong README).
- `scripts/export_yolo.py`: dataset YOLO cho một fold (`images/{train,val,cal,target}`, `labels/...`, `data.yaml`, `index.csv`); tên file = path thay `/` bằng `__`; `--link` (symlink) hoặc `--copy`; `--meta-root` chấp nhận bố cục repo (`data/manifest.csv`) và bố cục Kaggle (`manifest.csv` ngay gốc). Hàm `export_fold(..., limits=)` cho smoke test (nửa dương, nửa âm).
- `kaggle/train_fold.py`: export -> train YOLO11 (size/imgsz 640/AMP/seed/epochs/patience) -> val bằng best.pt -> predict val/cal/target (conf 0.001, iou 0.6, max_det 100, tất cả box) -> `preds_{fold}_seed{seed}.parquet` (orig_path, role, center, x1,y1,x2,y2, score; pixel ảnh gốc), `run_meta.json` (+ bản gắn tag), `index_{fold}.csv`, `best_{fold}_seed{seed}.pt`. Thời gian mỗi epoch lấy từ cột `time` (tích luỹ) của `results.csv`. `--smoke`: 1 epoch, 64 train / 16 ảnh mỗi role còn lại.
- `kaggle/README.md`: quy trình từng bước (dataset PRIVATE, cài đặt notebook, lệnh, file cần tải về `outputs/kaggle/`, cách ước tính ngân sách 6 fold x 3 seed).

## Kiểm tra (fold C4, `--copy`, 8037 ảnh, 14 s; script kiểm: scratchpad, không lưu trong repo)
- Mỗi role: số label == số ảnh; số dòng label == tổng `n_gt`: train 2063, val 346, cal 601, target 448 (khớp `docs/splits.md`). Mọi box trong [0,1], w,h > 0, không box nào bị bỏ sau khi clip (0 dropped).
- Round trip label -> pixel so với `gt_boxes` trên toàn bộ ảnh dương: sai số lớn nhất 0.001 px. 300 ảnh mẫu: kích thước thật == width/height trong manifest.
- 4 overlay (train/val/cal/target) xem bằng mắt: box xanh (từ label) trùng box đỏ (gt_boxes) và bao đúng polyp.
- Symlink thử được cả trên Windows (10 ảnh) — nhưng xem "Vấn đề mở" 3.
- Ánh xạ ngược dự đoán: assert `orig_shape` == (height, width) manifest cho mọi ảnh dự đoán; GT -> letterbox (LetterBox của Ultralytics, 640) -> `scale_boxes` (hàm predict dùng) -> so GT: lệch lớn nhất 1.33 px trên 40 ảnh (làm tròn pad/letterbox, ảnh lớn tới 1920 px). Parquet smoke: mọi box nằm trong ảnh, x2 > x1, đúng cột và dtype.

## Smoke run CPU (`--smoke`, C4, 64 train / 16 val / 16 cal / 16 target, yolo11s, 1 epoch)
Tổng 62 s thực (gồm tải trọng số); train 49.6 s (epoch 43.5 s gồm validation), predict 2.6/2.7/1.8 s cho 16 ảnh. mAP50 0.0004 (vô nghĩa, 1 epoch trên 64 ảnh). 1600 box mỗi role (16 ảnh x max_det 100: mạng chưa huấn luyện nên mọi ảnh đạt trần 100 box). **Không đại diện cho GPU.**

## Quy trình Kaggle
Xem `kaggle/README.md` (dataset private -> notebook GPU + internet -> `pip install ultralytics==8.4.170` -> smoke -> chạy thật -> tải `preds_*.parquet`, `run_meta*.json`, `index_*.csv`, `best_*.pt` về `outputs/kaggle/<fold>_seed<k>/`).

## Vấn đề mở
1. Siêu tham số huấn luyện (epochs 50, patience 10, batch 16, augment mặc định Ultralytics gồm mosaic và lật ngang) là mặc định chưa tinh chỉnh; cần chốt trước khi chạy cả 18 lần, và giữ cố định cho mọi fold/seed.
2. Ảnh không có box dự đoán nào không có dòng trong parquet; dùng `index_{fold}.csv` làm danh sách ảnh đầy đủ khi hiệu chuẩn.
3. Khi thử symlink bằng đường dẫn kiểu bash (`/c/Users/...`) truyền vào python Windows, một thư mục rác `C:\c\Users\...\linktest` (10 file) đã bị tạo ngoài dự án; lệnh xoá bị chặn bởi kiểm tra an toàn, cần người dùng xoá tay (`C:\c`). Không ảnh hưởng dữ liệu.
4. Chưa chạy trên Kaggle thật: thời gian GPU/epoch, `half=True` khi predict, và đường dẫn dataset Kaggle (có thể có thêm một cấp thư mục) cần xác nhận ở lần smoke đầu tiên. `data.yaml` ghi đường dẫn tuyệt đối của thư mục xuất.
5. Quota GPU hàng tuần và giới hạn thời gian mỗi phiên: người dùng tự kiểm tra trên Kaggle.
