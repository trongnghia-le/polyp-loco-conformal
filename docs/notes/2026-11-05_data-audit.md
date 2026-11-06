# 2026-11-05 — Kiểm tra dữ liệu PolypGen v3 và bảng kê từng khung

## Đã làm
- Kiểm tra zip PolypGen v3 (MD5 `bd0c0b6d70a0c0b1a5a36f6a5a200205`, chưa đối chiếu Synapse; zip đã giải nén vào `data/raw/`, bản zip nằm trong Thùng rác).
- Xác định trung tâm cho cả 23 chuỗi dương tính và 23 chuỗi âm tính: Fig. 7 của bài Sci Data 2023 + tổng số trong văn bản + giao diện máy nội soi in trên khung hình. Bài gốc có 2 lỗi hình (seq11 tô màu C3 nhưng là C1; Fig. 2a ghi âm tính C1/C4 = 223/1566, đúng là 430/1359). Chi tiết: `docs/data_audit.md`, script `scripts/audit_center_ui.py`.
- Bảng kê `data/manifest.csv` (8.037 dòng) do `scripts/build_manifest.py` tạo; báo cáo `docs/manifest_report.md`.

## Số liệu (từ `data/manifest.csv`)
> **Đính chính 2026-11-06:** bảng dưới là trước khi chuyển seq5 sang C2. Số đúng hiện tại: C2 = 301 / 1.115 / 0 / **1.416**, C5 = 208 / 0 / 731 / **939** (xem `docs/data_audit.md`, `docs/manifest_report.md`).

| Trung tâm | Ảnh đơn | Chuỗi dương | Chuỗi âm | Tổng |
|---|---|---|---|---|
| C1 | 256 | 228 | 430 | 914 |
| C2 | 301 | 865 | 0 | 1.166 |
| C3 | 457 | 0 | 0 | 457 |
| C4 | 227 | 450 | 1.359 | 2.036 |
| C5 | 208 | 250 | 731 | 1.189 |
| C6 | 88 | 432 | 1.755 | 2.275 |
| Tổng | 1.537 | 2.225 | 4.275 | 8.037 |

- Mask nhị phân ở ngưỡng 127; bỏ vùng < 50 px (`configs/data.yaml`, **tạm thời**). Lý do: 427 vùng < 200 px đều là vụn JPEG cạnh tổn thương lớn; polyp thật nhỏ nhất ~220 px.
- 3.121 khung có polyp, 4.916 không; 305 khung có > 1 polyp.
- Khung từ mask khớp bbox chính thức (IoU ≥ 0,5) ở 3.569/3.698 ảnh có file bbox; 129 lệch, phần lớn ở seq14 (mask vỡ 3–4 mảnh, bbox chính thức 1 khung) — **chưa điều tra**.
- C3: 64 ảnh có polyp nhưng thiếu file bbox (danh sách trong `docs/manifest_report.md`); ảnh `C3_EndoCV2021_00489].jpg` có dấu `]` thừa trong tên.
- Ảnh C2 `EndoCV2021_001010` (CSV ghi không có polyp, mask có pixel) là vụn 3×3 px → bị loại ở ngưỡng 50, khớp CSV.

## Quyết định
- Mask là nhãn gốc; khung tính từ mask; 1 polyp = 1 vùng liền (8-connectivity).
- C2, C3 không có chuỗi âm → cảnh báo giả chỉ đo ở C1, C4, C5, C6.

## Còn mở
- Mask vỡ mảnh ở seq14 (và một số seq5, seq13): có nên gộp các mảnh gần nhau thành 1 polyp? Ảnh hưởng trực tiếp tới định nghĩa "bỏ sót".
- Chuỗi dùng chung video gốc: chờ `data/clusters.csv`.

## Hình
- `results/figures/fig_dataset_composition.pdf` — Caption nháp: *Composition of the PolypGen dataset by centre. Bars show single frames, positive sequence frames and negative sequence frames; numbers give the total per centre. C2 and C3 contain no negative sequences and C3 no sequences at all.*
