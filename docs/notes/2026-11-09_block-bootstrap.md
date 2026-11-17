# 2026-11-09 — Block bootstrap và recalibration theo khối (vòng góp ý 3, M4), cùng các kiểm tra phụ

Script: `scripts/revision3_block_bootstrap.py` (chạy `.venv/Scripts/python.exe -I scripts/revision3_block_bootstrap.py`, ~45 s, CPU, chỉ dùng prediction đã lưu của 18 run YOLO11s; không train).
Output: `outputs/eval/revision3_block_bootstrap.csv`, `outputs/eval/revision3_block_recal.csv`, `outputs/eval/revision3_verdicts.csv`.
IoU ≥ 0.1, α = 0.10 (chỉ phân tích chính).

## Đã làm gì
Nhiều "video" ở trung tâm held-out là single frame không liên kết được (tỷ lệ positive frame là singleton: C5 0.98, C3 1.00, C2 0.21, C6 0.20 — `revision2_checks.csv` / `data/clusters.csv`). Bootstrap target-video coi chúng độc lập. Kiểm tra độ nhạy bằng cách gom single frame theo chỉ số file-name liên tiếp:
- `paper`: video cluster của paper (`data/clusters.csv`).
- `idx10`: khối 10 chỉ số liên tiếp (id // 10).
- `runs`: chuỗi chỉ số liên tiếp, cắt khi nhảy > 5 (định nghĩa của người góp ý, check2.py).
Frame của sequence folder giữ cluster; khối nào chung frame với một cluster thì gộp (connected components, scipy) để không cluster nào bị tách.

Số nhóm (positive target): C2 265/44/8, C3 456/51/2, C5 204/39/10, C6 88/19/10 (paper/idx10/runs).

Phần 1 — target bootstrap (C2, C3, C5, C6): ngưỡng cố định từ `revision2_R6a_weighted.csv` (frame equal, frame balanced_conservative, cluster equal, cluster balanced_plugin), resample nhóm (B = 1000), seed rút mỗi resample (`[2026, fold, 5152]` như R6a). Estimand không đổi (frame-avg và video-avg theo cluster của paper); chỉ đơn vị resample thô hơn.
Phần 2 — recalibration one frame × k đơn vị (C3, C5; k = 10, 20, cap n − 10): rút k khối, 1 frame dương ngẫu nhiên mỗi khối, CRC image-level, test trên frame của khối chưa rút (không khối nào nằm hai bên). Outcome theo miss rate trung bình theo khối (đại lượng thiết kế certify). `runs` quá ít khối (C3 2, C5 10) → không chạy được recal.
Phần 3 — bảng verdict (Table 3 main text): số weighting có mean > α (trong các weighting certify được), khoảng min–max, và vị trí khoảng bootstrap so với α cho 4 biến thể Table 2; cờ khi grouping thô hơn (≥ 8 nhóm) đổi verdict. Weighting: R6a (equal/conservative/plug-in), R6b (stratified, bỏ nếu degenerate ≥ 0.5), R6c (LOSO cluster), R3 (cap 1/5/20, bỏ từng centre, bỏ C2+C4; bỏ R3 "all" vì trùng R6a equal).

## Kiểm tra (assert trong script)
- Grouping `paper` tái tạo đúng `revision2_R6a_weighted.csv` (point, lo, hi, share > α) ở 128 ô (4 fold × 4 biến thể × 2 metric × 4 seed-row), sai khác < 1e-12 — cùng resample `[2026, fold, 5150]`.
- Recal với đơn vị `paper` tái tạo `conformal_cluster_summary.csv` fs_b (risk_cl, risk_fr, degenerate_frac, từng seed, k = 10, 20) và outcome held/violated/cannot certify của `revision_R4_degenerate_fa.csv` (16 kiểm tra).
- λ đọc từ R6a khớp lại risk_fr, risk_cl từng seed.

## Kết quả (seed all; `revision3_block_bootstrap.csv`)
| Fold | Grouping (nhóm) | Frame, all frames | Frame, centre-bal. | Video, all videos | Video, centre-bal. plug-in |
|---|---|---|---|---|---|
| C2 | paper (265) | 0.198 (0.061–0.289) | 0.180 (0.058–0.263) | 0.074 (0.045–0.109) | 0.070 (0.042–0.098) |
| C2 | idx10 (44) | (0.083–0.295) | (0.071–0.270) | (0.043–0.120) | (0.043–0.105) |
| C2 | runs (8) | (0.104–0.343) | (0.088–0.315) | (0.043–0.131) | (0.042–0.122) |
| C5 | paper (204) | 0.375 (0.305–0.446) | 0.312 (0.243–0.378) | 0.218 (0.154–0.283) | 0.262 (0.194–0.347) |
| C5 | idx10 (39) | (0.280–0.468) | (0.225–0.401) | (0.142–0.297) | (0.171–0.365) |
| C5 | runs (10) | (0.250–0.529) | (0.196–0.452) | (0.134–0.318) | (0.153–0.399) |
| C6 | paper (88) | 0.275 (0.144–0.432) | 0.192 (0.103–0.300) | 0.144 (0.078–0.221) | 0.178 (0.102–0.258) |
| C6 | idx10 (19) | (0.143–0.433) | (0.100–0.306) | (0.066–0.232) | (0.088–0.276) |
| C6 | runs (10) | (0.153–0.429) | (0.107–0.305) | (0.091–0.222) | (0.115–0.279) |
C3: không đổi verdict (runs chỉ 2 nhóm → không diễn giải).
C6 cận dưới chính xác: idx10 frame balanced 0.0998, video plug-in 0.0878 (`revision3_block_bootstrap.csv`).

Kết luận:
- **C5**: mọi khoảng vẫn > α ở mọi grouping (cận dưới ≥ 0.134; share > α ≥ 0.999). Kết luận mạnh nhất của paper đứng vững — khớp check của người góp ý (0.306–0.446 / runs 0.255–0.524 với RNG của họ).
- **C6**: frame-avg, all frames vẫn > α (cận dưới 0.143–0.153). Hai khoảng centre-balanced (0.103, 0.102) **chạm/cắt α** với idx10 (0.0998, 0.0878). → Bỏ claim "khoảng loại trừ α ở C6 cho cả hai đại lượng" khỏi Abstract; viết "mọi point estimate > α nhưng khoảng không loại trừ α nhất quán".
- **C2**: centre-balanced frame vẫn chứa α (0.071, 0.088). All-frames lên > α chỉ với 8 runs (0.104) — quá thô cho percentile interval, ghi nhận nhưng không dùng làm bằng chứng. Video plug-in từ "below" thành "includes" khi thô hơn.

Recal (`revision3_block_recal.csv`, k = 20, seed all):
| | cannot certify | held | violated | mean unit-avg | mean certified |
|---|---|---|---|---|---|
| C3 paper / idx10 | 0.044 / 0.048 | 0.671 / 0.650 | 0.285 / 0.302 | 0.078 / 0.080 | 0.081 / 0.083 |
| C5 paper / idx10 | 0.368 / 0.445 | 0.363 / 0.309 | 0.269 / 0.246 | 0.083 / 0.087 | 0.102 / 0.110 |
→ C3 gần như không đổi; C5 tỷ lệ không certify tăng 0.37 → 0.45, mean vẫn < α.

Verdict (`revision3_verdicts.csv`): C1 fr 13/13 (0.308–0.492), cl 6/7; C2 fr 12/12, cl 0/9; C3 fr 4/13 (0.036–0.133), cl 0/7; C4 fr 12/12, cl 6/7 (0.095–0.231); C5 fr 13/13 (0.217–0.411), cl 7/7 (0.181–0.262); C6 fr 13/13 (0.157–0.305), cl 7/7 (0.107–0.190).

## Các kiểm tra phụ cho vòng góp ý 3
- **M3, validation có dùng chọn checkpoint không?** Có. `scripts/export_yolo.py` ghi `data.yaml` với `val: images/val`; `kaggle/train_fold.py` gọi `model.train(data=..., patience=10)` → Ultralytics validate mỗi epoch trên split val, early stopping theo fitness val và lưu `best.pt` = epoch có fitness (0.1·mAP50 + 0.9·mAP50-95) cao nhất trên val (script tự tính lại best_epoch từ results.csv). Do đó loss trên val không độc lập với model → **không gộp val vào calibration** (không chạy phân tích merge). Số cluster dương trong val: C1 15, C2 8, C3 46, C4 1, C5 21, C6 9 (calibration: 57, 1, 91, 4, 40, 27) — tính từ `splits/fold_C*.csv` + `data/manifest.csv` (lệnh một dòng trong phiên, giống check1.py của người góp ý).
- **M6b**: in-domain video split: max +0.0427, 25/139 ô non-degenerate > α + 0.02 (`outputs/eval/run_conformal_3seed.log`, dòng "IN-DOMAIN CHECK, cluster-level splits").
- **M5b**: pooled, IoU ≥ 0.5, α = 0.10, frame-avg: C1 0.472, C2 0.173, C3 0.089, C4 0.274, C5 0.362, C6 0.311 (`conformal_summary.csv`, seed all).
- **Minor 3**: boot_frac_above_fr C1 = 0.835 (float lưu < 0.835 nên bảng S8 in 0.83) → text đổi thành "83–90%".
- **Minor 6**: `scripts/build_clusters.py` dòng 100 `assert not xc` (không còn cặp near-duplicate khác centre); docs/clusters.md.
- **Minor 8 (alarm/phút)**: readme.md của PolypGen v3 không nêu frame rate / khoảng lấy mẫu (grep fps/frame rate/second: không có) → không quy đổi được; ghi rõ trong Results.
- **Minor 14**: Crossref gan2027sdg: issued 2027-01, vol 115 → giữ năm 2027 (`paper/citation_check.md`, Round 3).

## Hình
`scripts/fig_cluster_budget.py` sửa: bỏ các budget bị cap (C4: k = 40 → 24 trùng gần k = 20); giá trị vẫn ở Supplementary Table S15. Tái tạo `results/figures/fig_cluster_budget.{pdf,eps,png}`, đã xem PNG ở 183 mm: C4 còn k = 5, 10, 20, không chồng điểm.
Caption nháp: "Recalibration at the held-out centre with the labelling budget counted in videos; at C4 the capped k = 40 (24 videos) is omitted (Supplementary Table S15)."
Không tạo hình mới cho block bootstrap (đưa vào Supplementary Table S17 và main Table 3 dạng bảng).

## Thay đổi claim trong manuscript
- Abstract: bỏ "intervals excluded α at C5 and C6 for both quantities and at C4 for the video average"; thay bằng C5 (mọi weighting, khoảng > α, kể cả grouping thô hơn) và C6 (mọi point > α, khoảng không loại trừ α nhất quán). Bỏ so sánh frame-avg của recalibration khỏi Abstract; nêu rằng 0.078–0.085 là điều exchangeability dự đoán.
- Display items: 4 bảng + 3 hình = 7 (thêm Table 3 verdict; recalibration thành Table 4).

## Vấn đề mở
- File-name order chỉ là proxy cho thứ tự ghi hình; kết quả block chỉ là độ nhạy.
- Không chạy M1(b) (thiết kế nhắm frame-avg), M2(c) (jackknife video C2), minor 16 (gom sequence ClinicDB theo study) — optional.
