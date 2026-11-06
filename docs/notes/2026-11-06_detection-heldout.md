# 2026-11-06 — Chất lượng detection: trong miền vs trung tâm bị loại (seed 0, 1)

Script: `scripts/eval_detection.py` → `outputs/eval/detection_summary.csv`. Khớp polyp ở IoU ≥ 0,5, một-đổi-một, tham lam theo điểm (như `docs/metric.md`). Tự kiểm: AP50 trên val của mình khớp mAP50 của Ultralytics (sai khác < 0,06) ở cả 12 lần chạy.

## Số liệu (trung bình 2 seed)
| Trung tâm | AP50 val (trong miền) | AP50 target (bị loại) | Bỏ sót @ conf 0,25 target | Recall tối đa target (mọi box ≥ 0,001) |
|---|---|---|---|---|
| C1 | 0,706 | **0,489** | 53 % | 0,753 |
| C2 | 0,696 | 0,704 | 33 % | 0,857 |
| C3 | 0,708 | 0,844 | 23 % | 0,940 |
| C4 | 0,796 | **0,465** | 36 % | 0,874 |
| C5 | 0,754 | **0,545** | 48 % | 0,827 |
| C6 | 0,726 | 0,703 | 36 % | 0,879 |

- Sụt mạnh khi đổi bệnh viện ở C1, C4, C5 (AP50 giảm 0,21–0,33); C2, C6 gần như giữ nguyên; C3 cao hơn trong miền. Khớp mẫu hình đã công bố (C4 khó nhất, C3 dễ nhất: Corbetta 2023, "Metrics or Mirage" 2026) → kết quả đáng tin.
- Cảnh báo giả trên khung âm tính (conf ≥ 0,25): 0,07–0,28 box/khung ở C1, C4, C5, C6 (C2, C3 không có chuỗi âm).

## Phát hiện quan trọng cho thiết kế conformal: "sàn bỏ sót"
Recall tối đa (giữ mọi box, conf ≥ 0,001; trung bình 2 seed) theo tiêu chí IoU:

| IoU | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| target 0,1 | 0,903 | 0,911 | 0,953 | 0,942 | 0,920 | 0,931 |
| target 0,3 | 0,834 | 0,884 | 0,949 | 0,927 | 0,882 | 0,906 |
| target 0,5 | 0,753 | 0,857 | 0,940 | 0,874 | 0,827 | 0,879 |
| val 0,5 | 0,929 | 0,900 | 0,880 | 0,899 | 0,871 | 0,875 |

→ Không ngưỡng λ nào làm tỷ lệ bỏ sót thấp hơn 1 − recall tối đa. Với IoU ≥ 0,5, α = 0,05 và 0,10 **không thể đạt** ở nhiều trung tâm (sàn 6–25 % ở target, 7–13 % ngay cả trong miền); CRC sẽ trả λ thoái hoá. Với IoU ≥ 0,1 sàn còn 5–10 %: α = 0,10 khả thi gần như mọi nơi, α = 0,05 vẫn không.

## Quyết định cần người dùng chốt (chưa chạy conformal)
1. Tiêu chí "phát hiện được": giữ IoU ≥ 0,5 (định vị) hay chuyển sang IoU ≥ 0,1 / chạm polyp (lâm sàng: cảnh báo có đè lên polyp); báo cái còn lại làm phân tích độ nhạy. Lưu ý minh bạch: thay đổi sau khi đã thấy sàn bỏ sót → phải ghi rõ trong bài.
2. Lưới α: {0,05; 0,10; 0,20} (kế hoạch) hay {0,10; 0,15; 0,20; 0,30}; α không khả thi được báo như một kết quả ("bảo đảm chỉ đạt được trong giới hạn recall của detector").

## Hình
Chưa vẽ — chờ seed 2 để có số cuối (3 seed). Ứng viên: dumbbell AP50 trong miền vs trung tâm bị loại theo từng trung tâm, điểm từng seed.

## Cập nhật: thử YOLO11m (2026-11-06 ~17:30)
Người dùng đồng ý thử detector lớn hơn ("kết quả thế nào cũng báo cáo được"). Notebook `run_mpilot` (YOLO11m, seed 0, C1 và C4, cùng tham số 50 epoch / patience 10 / 640 px) đang chạy song song với seed 2. Tiêu chí quyết định: nếu sàn bỏ sót ở trung tâm bị loại (1 − recall tối đa, IoU 0,1 và 0,5) giảm rõ (ví dụ C1 từ ~10 % xuống < 6 % ở IoU 0,1) → chạy đủ 18 lần với YOLO11m bằng quota tuần sau, YOLO11s thành ablation; nếu không → giữ YOLO11s, ghi "mô hình lớn hơn không hạ sàn" làm kết quả.
