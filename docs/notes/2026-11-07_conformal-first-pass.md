# 2026-11-07 — Conformal risk control, lượt đầu (YOLO11s, seed 0–1)

Script: `scripts/conformal.py` (chạy: `.venv/Scripts/python.exe -I scripts/conformal.py --root outputs/kaggle`, ~3 phút CPU) →
`outputs/eval/conformal_long.parquet` (276 240 dòng: run × IoU × α × chiến lược × ngân sách × lần rút/resample) và
`outputs/eval/conformal_summary.csv` (gộp theo seed; dòng `seed=all` và từng seed). Mọi số dưới đây lấy từ
`conformal_summary.csv` (dòng `seed=all` trừ khi ghi khác) hoặc từ stdout của script. Hình: `scripts/fig_risk_vs_alpha.py`,
`scripts/fig_fewshot_budget.py` (đọc lại từ `conformal_summary.csv`).

## Đã chạy gì
- 12 run có sẵn: C1–C6 × seed 0, 1 (seed 2 chưa về; YOLO11m sau — chạy lại với `--root outputs/kaggle_m`, file ra có hậu tố `_kaggle_m`).
- Cỡ (stdout, giống nhau giữa 2 seed): ảnh dương cal nguồn / ảnh dương đích / ảnh âm đích = C1 541/387/527, C2 367/1233/183, C3 528/456/1, C4 543/432/1604, C5 578/206/733, C6 538/407/1868.
- Loss theo `docs/metric.md`: L_i(λ) = phân số polyp bị bỏ sót ở ngưỡng điểm λ, ghép một-một tham lam theo điểm; IoU ≥ 0,1 (chính) và ≥ 0,5 (độ nhạy); lưới λ 0,001…0,999; α ∈ {0,05; 0,10; 0,15; 0,20; 0,30}; λ̂ = λ lớn nhất có (ΣL_i + 1)/(n + 1) ≤ α; rỗng → λ̂ = 0,001, đánh dấu degenerate. Chỉ ảnh dương vào hiệu chuẩn và rủi ro.
- Chiến lược với trung tâm đích T:
  - **in-domain** (tham chiếu): 200 lần chia 50/50 *theo cụm* tập cal nguồn, hiệu chuẩn nửa này, đánh giá nửa kia. Thêm **in-domain_image**: 200 lần chia 50/50 *theo ảnh* (bỏ qua cụm) — chỉ để kiểm lỗi code (trao đổi được chính xác).
  - **pooled**: λ̂ trên toàn bộ ảnh dương cal nguồn, đánh giá trên toàn bộ ảnh đích.
  - **worst-case**: λ̂_c trên cal của từng trung tâm nguồn, dùng min_c λ̂_c; ghi trung tâm "binding".
  - **\*_boot**: 200 bootstrap cụm của cal nguồn (lấy lại có hoàn lại các cụm *có ảnh dương*, phân tầng theo trung tâm để trung tâm nào cũng còn n_c > 0), tính lại λ̂ và rủi ro đích → tần suất vi phạm = độ bất ổn của bảo đảm.
  - **few-shot**: 500 lần rút đầu tiên trong `splits/target_draws_T.json` cho ngân sách 20/50/100 (`--draws`); hiệu chuẩn trên ảnh dương của cụm được rút, đánh giá trên cụm đích còn lại. k_cal = số cụm có ≥ 1 ảnh dương trong lần rút.
  - **shift-weighted: TODO** — chưa làm vì không lưu đặc trưng ảnh (cần embedding/score theo ảnh từ Kaggle để ước lượng trọng số likelihood-ratio).
- Đầu ra mỗi dòng: λ̂, degenerate, λ̂ = 0,999, rủi ro thực (trung bình L_i trên ảnh dương test), tỷ lệ bỏ sót theo polyp (micro), vi phạm (rủi ro > α), cảnh báo giả/khung âm (chỉ đích C1, C4, C5, C6; NaN ở C2, C3), FP/khung dương (box không ghép có score ≥ λ̂), n_cal, k_cal, n_test.

## Kiểm tra (sanity checks)
1. **Tính chất tiền tố**: L_i(λ) tính từ điểm của box đã ghép khớp chính xác `metrics.miss_fraction` trên 200 ảnh dương ngẫu nhiên × 12 giá trị λ × 2 IoU × 12 run (stdout: "prefix property ... OK"); không cần fallback. Thêm assert L_i không giảm theo λ.
2. **λ̂ vectorized = `metrics.crc_lambda`** (assert ở chiến lược pooled, mọi run × IoU × α).
3. **In-domain, chia theo ảnh (trao đổi được)**: max(trung bình rủi ro − α) trên các ô không degenerate = **+0,003** → ĐẠT (CRC + code đúng).
4. **In-domain, chia theo cụm**: max = **+0,038**; 18/93 ô vượt α + 0,02 — tập trung ở fold C3 (cả hai seed, mọi α), và lẻ tẻ C1, C4, C5, C6. *Không phải lỗi code* (mục 3 đạt); nguyên nhân là đơn vị trao đổi là cụm, cụm rất lớn: cal C2 chỉ là **một cụm** `C2_seq12` (252 ảnh dương), `C4_vp0` 60, `C6_seq19` 55. Ở fold C3, cal không có 91 ảnh đơn C3 để pha loãng, nên ~half khối lượng hiệu chuẩn nằm trong vài cụm; khi cụm khó rơi vào nửa test thì rủi ro test vọt lên, khi rơi vào nửa cal thì λ̂ thấp nhưng phần test dễ — trung bình theo ảnh lệch dương. Đây chính là cảnh báo ở `docs/metric.md` §6 ("bảo đảm chỉ gần đúng"), nay có số đo.
5. Tần suất vi phạm in-domain ≈ 0,43–0,55 — đúng kỳ vọng: CRC bảo đảm *kỳ vọng*, không phải P(rủi ro ≤ α).

## Kết quả chính (IoU ≥ 0,1; rủi ro thực trung bình, trong ngoặc tần suất vi phạm)
In-domain/few-shot: trung bình qua lần chia/rút × 2 seed. Pooled/worst-case: ước lượng điểm trung bình 2 seed (vi phạm = tỷ lệ trong 200 bootstrap × 2 seed).

α = 0,10:

| Đích | in-domain | pooled | worst-case | few-shot 20 | few-shot 50 | few-shot 100 |
|---|---|---|---|---|---|---|
| C1 | 0,112 (0,47) | 0,478 (1,00) | 0,133 (0,55) | 0,278 (0,87) | 0,305 (0,81) | 0,257 (0,55) |
| C2 | 0,096 (0,43) | 0,196 (1,00) | 0,077 (0,20) | 0,171 (0,60) | 0,168 (0,52) | 0,149 (0,38) |
| C3 | 0,132 (0,52) | 0,084 (0,10) | 0,017 (0,00) | 0,077 (0,27) | 0,092 (0,37) | 0,095 (0,41) |
| C4 | 0,114 (0,50) | 0,319 (1,00) | 0,064 (0,10) | 0,121 (0,37) | 0,126 (0,43) | 0,106 (0,47) |
| C5 | 0,116 (0,52) | 0,383 (1,00) | 0,146 (0,70) | 0,091 (0,29) | 0,083 (0,27) | 0,086 (0,30) |
| C6 | 0,105 (0,51) | 0,240 (1,00) | 0,095 (0,48) | 0,138 (0,52) | 0,137 (0,57) | 0,123 (0,63) |

α = 0,20:

| Đích | in-domain | pooled | worst-case | few-shot 20 | few-shot 50 | few-shot 100 |
|---|---|---|---|---|---|---|
| C1 | 0,189 (0,46) | 0,557 (1,00) | 0,296 (0,74) | 0,414 (0,89) | 0,413 (0,81) | 0,324 (0,55) |
| C2 | 0,198 (0,43) | 0,318 (1,00) | 0,169 (0,13) | 0,285 (0,72) | 0,264 (0,60) | 0,231 (0,43) |
| C3 | 0,226 (0,52) | 0,183 (0,19) | 0,032 (0,00) | 0,176 (0,34) | 0,193 (0,44) | 0,197 (0,46) |
| C4 | 0,201 (0,50) | 0,482 (1,00) | 0,185 (0,42) | 0,222 (0,42) | 0,229 (0,42) | 0,221 (0,49) |
| C5 | 0,205 (0,52) | 0,505 (1,00) | 0,342 (0,92) | 0,176 (0,36) | 0,185 (0,38) | 0,192 (0,41) |
| C6 | 0,208 (0,52) | 0,362 (1,00) | 0,175 (0,29) | 0,239 (0,47) | 0,229 (0,53) | 0,212 (0,56) |

Đọc:
- **Pooled CRC hỏng khi đổi bệnh viện** ở 5/6 trung tâm: ở α = 0,10 rủi ro thực 0,20–0,48 (gấp 2–5 lần α), vi phạm ở 100 % bootstrap; chỉ C3 (trung tâm "dễ", AP50 đích > trong miền) giữ được (0,084). λ̂ pooled (α = 0,10) 0,05–0,29.
- **Worst-case** bảo thủ hơn hẳn (λ̂ trung vị 0,001–0,010 ở α = 0,10) và đạt α ở C2, C3, C4, C6, nhưng vẫn vượt ở C1 (0,133) và C5 (0,146; α = 0,20: 0,342). Trung tâm binding gần như luôn là **C5** (cal C5 chỉ 41 ảnh dương, mỗi ảnh một cụm); khi đích là C5 thì C6/C1 binding. Cái giá: cảnh báo giả rất cao — C1 8,9 box/khung âm, C4 4,0, C5 1,2, C6 0,53 ở α = 0,10 (pooled: 0,16 / 0,26 / 0,08 / 0,02). Ở α = 0,10, IoU 0,1 fold C2 worst-case degenerate cả 2 seed (cal C5 có > 3 polyp bỏ sót ngay ở λ = 0,001).
- **Few-shot đích** sửa được dịch chuyển trung bình ở C3, C4, C5, C6 (rủi ro ~α), nhưng **không ở C1** (0,26–0,31 ở α = 0,10) và chỉ một phần ở C2. Nguyên nhân C1: cụm `C1_seq11` chứa 136/387 ảnh dương đích và rất khó; seed 0, ngân sách 50: 403/500 lần rút không chứa seq11 → rủi ro test trung bình 0,418 (seq11 chiếm phần lớn tập test); 97 lần chứa seq11 → toàn bộ degenerate, rủi ro test 0,020. Cùng cơ chế cụm như mục kiểm tra 4, nhưng ở đích mạnh hơn nhiều. C2 tương tự (các cụm seq5/12–15 lớn; ở ngân sách 100, 91 % lần rút có một cụm ≥ ngân sách, xem `docs/splits.md`).
- Tăng ngân sách 20 → 100 không đưa tần suất vi phạm về mức in-domain một cách đơn điệu (C1, C2 giảm; C3, C4, C6 tăng nhẹ lên ~mức in-domain 0,5) — vì vi phạm ~0,5 là mức "bình thường" khi rủi ro trung bình ≈ α; giảm vi phạm dưới 0,5 chỉ xảy ra khi λ̂ bảo thủ (n nhỏ, nhiều lần degenerate).
- IoU ≥ 0,5 (độ nhạy, α = 0,10): pooled 0,175–0,473 (C3 0,077), worst-case 0,026–0,239; nhiều degenerate hơn hẳn (xem dưới).

## Degenerate (λ̂ = 0,001)
Tỷ lệ trung bình qua fold (stdout bảng "DEGENERATE fraction"; `degenerate_frac` trong summary):
- IoU 0,1: pooled 0 ở mọi α; in-domain 0,05 ở α = 0,05, 0 từ α ≥ 0,10; worst-case điểm 0,92 (α = 0,05), 0,17 (α = 0,10), 0 từ 0,15; few-shot ở α = 0,10: 0,20 / 0,15 / 0,15 (ngân sách 20/50/100), ở α = 0,05: 0,45–0,57.
- IoU 0,5: pooled 0,17 ở α = 0,05; in-domain 0,48 (α = 0,05); worst-case 1,00 / 0,83 / 0,58 ở α = 0,05 / 0,10 / 0,15; few-shot 0,44–0,50 ở α = 0,10.
- Few-shot α = 0,10, IoU 0,1 theo fold: C2 0,26–0,42 và C5 0,44 (ngân sách 20) cao nhất. Degenerate ở few-shot kéo rủi ro xuống sàn bỏ sót (λ̂ = 0,001) — tức "an toàn nhờ thất bại", kèm cảnh báo giả rất cao.
→ α = 0,05 về cơ bản không khả thi (đúng như dự báo trong `2026-11-06_detection-heldout.md`); IoU 0,5 với α ≤ 0,10 cũng vậy.

## Lưu ý / hạn chế
- Mới 2 seed; seed 2 tối nay → chạy lại cùng lệnh, file bị ghi đè. Error bar trong hình = khoảng (min–max) 2 seed.
- Shift-weighted CRC chưa làm (TODO ở trên).
- Bootstrap phân tầng theo trung tâm và chỉ lấy cụm có ảnh dương (lệch nhẹ so với "resample clusters" thuần, để trung tâm nhỏ không mất hết ảnh dương; bản đầu lấy cả cụm chỉ-âm làm C4 — 5 cụm cal, phần lớn âm — thường không còn ảnh dương nào, gây worst-case degenerate giả 31 % ngay ở α = 0,30). Sau sửa, worst-case bootstrap vẫn degenerate khá nhiều ở α = 0,10, IoU 0,1: C1 0,26, C2 0,64, C3 0,25, C4 0,08, C5 0,23, C6 0,16 (α = 0,20: ≤ 0,06) — degenerate đẩy rủi ro đích xuống sàn, nên tần suất vi phạm worst-case ở α = 0,10 bị "làm đẹp" một phần bởi thất bại. C2 trong cal chỉ có một cụm dương nên λ̂_C2 không đổi giữa các bootstrap.
- Fold C2, C3: FA trên khung âm để NaN theo quy ước (không có chuỗi âm).
- Trung bình rủi ro in-domain theo cụm có thể > α (đến +0,038) — đây là giới hạn thật của bảo đảm khi khung cùng video, phải nêu trong bài; bản chia theo ảnh là kiểm tra code, không phải kết quả báo cáo.
- `docs/metric.md` §7 dự kiến R = 1000 lần rút; ở đây dùng 500 đầu (`--draws 1000` để chạy đủ).

## Hình
- `results/figures/fig_risk_vs_alpha.{pdf,eps,png}` (183 × 108 mm). Draft caption: *Realized miss rate on the held-out hospital versus the target miss rate α (IoU ≥ 0.1) for six leave-one-centre-out folds (a–f). In-domain: CRC calibrated and evaluated on disjoint cluster halves of the source calibration set (mean of 200 splits); pooled source: one threshold from all source calibration images; per-source worst case: lowest of the per-centre thresholds; target few-shot: calibrated on ≈50 positive target images (mean of 500 cluster draws). Points are means over two training seeds; error bars show the range across seeds. Dashed line, y = α.*
- `results/figures/fig_fewshot_budget.{pdf,eps,png}` (183 × 62 mm). Draft caption: *Frequency with which the realized target miss rate exceeds α after few-shot recalibration on 20, 50 or 100 positive target images (500 cluster draws per budget), at α = 0.10 (a) and α = 0.20 (b), IoU ≥ 0.1. Dashed segments, violation frequency of in-domain CRC (cluster split of the source calibration set). Bars are pooled over two seeds; error bars show the range across seeds.*
