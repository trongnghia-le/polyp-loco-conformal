# 2026-11-08 — Kết quả 3 seed (YOLO11s, 18/18 lần chạy)

Chạy lại: `scripts/eval_detection.py`, `scripts/conformal.py --root outputs/kaggle`, `scripts/conformal_cluster.py --root outputs/kaggle`, các `scripts/fig_*.py` (log: `outputs/eval/run_*_3seed.log`). Số liệu 3 seed thay thế số 2 seed trong các ghi chú 2026-11-07.

## Conformal theo ảnh (IoU ≥ 0,1, α = 0,10; nguồn `outputs/eval/conformal_summary.csv`, log `run_conformal_3seed.log`)
Rủi ro thực tế ở trung tâm bị loại, trung bình 3 seed (tần suất vi phạm):
| | trong miền | gộp nguồn | an toàn nhất | +20 ảnh | +50 ảnh | +100 ảnh |
|---|---|---|---|---|---|---|
| C1 | 0,116 (0,48) | 0,468 (1,00) | 0,167 (0,70) | 0,289 (0,88) | 0,316 (0,81) | 0,264 (0,55) |
| C2 | 0,096 (0,43) | 0,198 (1,00) | 0,080 (0,24) | 0,174 (0,62) | 0,168 (0,51) | 0,148 (0,34) |
| C3 | 0,135 (0,50) | 0,100 (0,36) | 0,020 (0,00) | 0,077 (0,26) | 0,091 (0,36) | 0,095 (0,40) |
| C4 | 0,110 (0,50) | 0,288 (1,00) | 0,051 (0,07) | 0,117 (0,35) | 0,122 (0,41) | 0,109 (0,47) |
| C5 | 0,113 (0,51) | 0,375 (1,00) | 0,137 (0,79) | 0,090 (0,30) | 0,085 (0,29) | 0,087 (0,31) |
| C6 | 0,108 (0,48) | 0,275 (1,00) | 0,087 (0,37) | 0,149 (0,55) | 0,142 (0,59) | 0,126 (0,59) |
Kiểm tra trong miền: chia theo ảnh max +0,0034 → ĐẠT; chia theo cụm max +0,0427 (25/139 ô) → cờ (đúng như giải thích: khung cùng video không hoán đổi được).

## Conformal theo cụm (nguồn `conformal_cluster_summary.csv`, log `run_cluster_3seed.log`)
α = 0,10, rủi ro trung bình theo cụm / theo khung:
- Gộp nguồn theo cụm: C1 0,113/0,358; C2 0,074/0,188; C3 0,037; C4 0,174/0,148; C5 0,218/0,225; C6 0,144/0,157 → vẫn vỡ ở C4, C5, C6.
- k = 20 video, (a) CRC theo cụm / (b) 1 khung mỗi video: rủi ro theo cụm ≤ 0,085 ở mọi trung tâm; (c) mọi khung của video: C4 0,117, C6 0,100. Chi phí gán nhãn (c) trung vị 686 khung ở C4, 121 ở C6; (b) = 20 khung.
- Theo khung, C1 vẫn 0,25–0,29 với mọi thiết kế.

## Detector (`detection_summary.csv`, `max_recall.csv`)
Recall tối đa ở trung tâm bị loại, IoU 0,1 / 0,5: C1 0,890/0,750; C2 0,907/0,855; C3 0,953/0,937; C4 0,949/0,883; C5 0,927/0,824; C6 0,925/0,874.

## Hình (đã vẽ lại, đã xem ở kích thước thật)
`fig_risk_vs_alpha`, `fig_fewshot_budget`, `fig_cluster_budget` — sai số = khoảng giữa các seed (ghi trong caption).

> **Đính chính 2026-11-08:** gộp nguồn theo cụm ở α = 0,10 cũng vượt ở **C1** (rủi ro trung bình theo video 0,113 > 0,10, `conformal_cluster_summary.csv`), không chỉ C4–C6. Gộp nguồn theo ảnh ở C3 nằm sát α (trung bình 0,100; seed 2 = 0,132; 36 % lần lấy mẫu lại vi phạm). Hiệu chuẩn lại theo khung cũng chưa đạt ở C6 (0,126–0,149).
