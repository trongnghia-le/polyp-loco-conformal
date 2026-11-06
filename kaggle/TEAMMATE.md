# Chạy YOLO11m trên tài khoản Kaggle của thành viên nhóm

## Chủ dataset (letronghia) làm một lần
1. Mở dataset `polypgen-loco` trên Kaggle → **Settings** (hoặc nút **Share**) → **Add collaborators** → nhập username Kaggle của bạn cùng nhóm, quyền **Can view**. Dataset vẫn để **Private**.
2. Gửi cho bạn ấy 5 file notebook trong thư mục `kaggle/` của repo: `run_m_seed0_rest.ipynb`, `run_m_seed1_a.ipynb`, `run_m_seed1_b.ipynb`, `run_m_seed2_a.ipynb`, `run_m_seed2_b.ipynb`.
3. Dữ liệu PolypGen có điều khoản sử dụng trên Synapse: bạn cùng nhóm nên tự đăng ký Synapse và chấp nhận điều khoản PolypGen trước khi dùng.

## Thành viên nhóm làm với từng notebook
1. Kaggle → **Code** → **New Notebook** → **File → Import Notebook** → chọn file `.ipynb`.
2. Cột phải **Add Input** → tab **Shared with you** (hoặc tìm `polypgen-loco`) → thêm dataset.
3. **Settings**: **Accelerator = GPU** (T4), **Internet = On**. Tài khoản cần xác minh số điện thoại để dùng GPU.
4. **Save Version → Save & Run All (Commit)**. Notebook chạy nền, tắt máy được. Không cần sửa dòng nào.
5. Kaggle chỉ cho **2 notebook GPU chạy nền cùng lúc**: chạy 2 cái trước, khi một cái xong thì chạy cái tiếp theo.
6. Khi xong: mở bản commit → tab **Output** → tải file `results_m_seed..._....zip` → gửi lại cho chủ dự án (đặt vào `outputs/kaggle_m/`).

| Notebook | Vòng | Ước tính |
|---|---|---|
| `run_m_seed0_rest` | seed 0: C2, C3, C5, C6 | ~5–6,5 giờ |
| `run_m_seed1_a` | seed 1: C1, C2, C3 | ~4–5 giờ |
| `run_m_seed1_b` | seed 1: C4, C5, C6 | ~4–5 giờ |
| `run_m_seed2_a` | seed 2: C1, C2, C3 | ~4–5 giờ |
| `run_m_seed2_b` | seed 2: C4, C5, C6 | ~4–5 giờ |

Tổng khoảng 22–27 giờ GPU (ước tính trước khi có số đo của bản thử YOLO11m). Seed 0 C1 và C4 đã chạy ở tài khoản chủ (`run_mpilot`).

## Nếu có lỗi
Notebook tự dừng sau 2 lần chạy lỗi liên tiếp và không bắt đầu lần chạy mới sau 8 giờ. Nếu thấy chữ `FAILED` trong log hoặc notebook dừng sớm: chụp tab Logs gửi lại, đừng chạy lại ngay.
