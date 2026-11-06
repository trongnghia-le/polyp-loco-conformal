# 2026-11-06 — Vòng C4 seed 0 trên Kaggle; notebook chạy tất cả

- Dataset private `polypgen-loco` (8.037 ảnh) đã lên Kaggle; hai ảnh tên lạ được đổi tên trong bản tải lên (`rename_map.csv`).
- Huấn luyện C4 seed 0: 50 epoch, 2.810 s (~47 phút, ~56 s/epoch), không dừng sớm. Validation (trung tâm nguồn): mAP50 0,798, mAP50-95 0,655, precision 0,90, recall 0,73 (từ `results.csv`, epoch 50).
- Dự đoán lần đầu hết bộ nhớ GPU (trainer giữ ~12 GB) → sửa `train_fold.py`: giải phóng bộ nhớ, `--pred-batch 4`, `--skip-train`.
- Ước tính 18 lần chạy ≈ 15–17 giờ GPU; quota còn 29 giờ (theo người dùng).
- `kaggle/run_seed{0,1,2}.ipynb`: ba notebook độc lập, mỗi seed 6 lần chạy (~5–6 giờ), không cần sửa gì; C4 seed 0 chạy tương tác chỉ để đo thời gian và được chạy lại trong notebook cho đồng nhất phiên bản code.
- Chưa kiểm chứng trên Kaggle: notebook mới chỉ được kiểm cú pháp cục bộ.
