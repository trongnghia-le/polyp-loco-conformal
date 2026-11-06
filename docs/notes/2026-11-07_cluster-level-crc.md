# 2026-11-07 — CRC cấp cụm (video là đơn vị trao đổi được) và ngân sách gán nhãn tính theo video

Script: `scripts/conformal_cluster.py` (chạy: `.venv/Scripts/python.exe -I scripts/conformal_cluster.py --root outputs/kaggle`,
~13 phút CPU; `--root outputs/kaggle_m` cho YOLO11m, file ra có hậu tố `_kaggle_m`). Import loading / loss theo matched
score / `crc_idx` / `evaluate` từ `scripts/conformal.py` (không sửa file đó). Đầu ra:
`outputs/eval/conformal_cluster_long.parquet` (1 020 360 dòng: run × IoU × α × chiến lược × k/ngân sách × lần rút) và
`outputs/eval/conformal_cluster_summary.csv` (dòng `seed=all` + từng seed). Mọi số dưới đây lấy từ
`conformal_cluster_summary.csv` (`seed=all`, IoU 0,1 trừ khi ghi khác) hoặc stdout của script.
Hình: `scripts/fig_cluster_budget.py` → `results/figures/fig_cluster_budget.{pdf,eps,png}`.

## Lý do
Lượt đầu (`2026-11-07_conformal-first-pass.md`): (i) in-domain, CRC theo ảnh với chia 50/50 *theo cụm* vượt α tới +0,038
(fold C3), trong khi chia theo ảnh thì chính xác → khung cùng video không trao đổi được; (ii) few-shot đích không cứu được
C1/C2 vì ảnh dương đích dồn vào vài video dài (`C1_seq11` = 136/387). Giả thuyết: nếu lấy **cụm** (`cluster_id`,
`data/clusters.csv`) làm đơn vị trao đổi được thì bảo đảm CRC đúng trở lại, và ngân sách gán nhãn nên đếm bằng video.

## Định nghĩa chính xác
- Ảnh dương: `n_gt > 0` (`gt_boxes` trong `data/manifest.csv`, khoá qua `clean()` như `conformal.py`). L_i(λ) như `docs/metric.md` §3.
- Đơn vị: cụm có ≥ 1 ảnh dương. **Loss cụm** L_c(λ) = trung bình L_i(λ) trên m_c ảnh dương của cụm ∈ [0, 1], không giảm theo λ.
- **CRC cấp cụm**: λ̂ = sup{λ ∈ Λ : (Σ_c L_c(λ) + 1)/(n_c + 1) ≤ α}, n_c = số cụm hiệu chuẩn; rỗng → λ̂ = 0,001, degenerate.
  Cỡ tối thiểu n_c ≥ 1/α − 1 cụm (α = 0,10 → 9; 0,20 → 4).
- **Bảo đảm** (giả sử cụm hiệu chuẩn và cụm mới trao đổi được): E[L_{c_new}(λ̂)] ≤ α, tức bảo đảm cho **rủi ro trung bình
  theo cụm** (mỗi video đếm một lần, bất kể dài bao nhiêu khung) — *không* phải rủi ro trung bình theo khung mà lượt đầu
  báo. Hệ quả: một video dài, khó (C1_seq11) chỉ nặng bằng một ảnh đơn; rủi ro theo khung có thể vẫn > α.
- Báo cả hai: `risk_cl` (trung bình L_c trên cụm dương test — cái được bảo đảm) và `risk_fr` (trung bình L_i trên ảnh dương
  test — cái lượt đầu báo); thêm `micro_miss` (Σ bỏ sót / Σ polyp), `fa_neg` (box/khung âm, chỉ đích C1, C4, C5, C6),
  `viol_cl` / `viol_fr` = risk > α.
- Chiến lược (đích T):
  1. `in_domain_cl`: 200 lần chia 50/50 các cụm dương của cal nguồn; CRC cấp cụm trên nửa này, test nửa kia.
     `in_domain_img`: cùng các lần chia, CRC theo ảnh trên mọi ảnh dương của nửa hiệu chuẩn (so sánh).
  2. `pooled_cl`: CRC cấp cụm trên toàn bộ cụm dương cal nguồn → toàn bộ đích; `pooled_cl_boot`: 200 bootstrap cụm dương
     phân tầng theo trung tâm (vi phạm = tỷ lệ bootstrap).
  3. `worst_cl`: min λ̂ cấp cụm theo từng trung tâm nguồn; `worst_cl_feas`: chỉ lấy min trên các trung tâm có
     1/(n_c + 1) ≤ α (khả thi theo cỡ).
  4. Few-shot theo cụm, k ∈ {5, 10, 20, 40} cụm dương đích rút đều không hoàn lại, **chặn ở (số cụm dương đích − 10)** để
     test còn ≥ 10 cụm. Chặn (stdout): C1 148, C2 255, C3 446, **C4 24** (k = 40 → 24), C5 194, C6 78.
     (a) `fs_a`: CRC cấp cụm trên k cụm; (b) `fs_b`: CRC theo ảnh trên **một** ảnh dương ngẫu nhiên mỗi cụm rút;
     (c) `fs_c`: CRC theo ảnh trên mọi ảnh dương của k cụm (= few-shot hiện tại nhưng đếm bằng cụm).
     Test = mọi cụm đích không được rút. `fs_frames`: các lần rút ngân sách ảnh 20/50/100 đã lưu (`splits/target_draws_*.json`,
     500 lần đầu), CRC theo ảnh như `conformal.py`, đánh giá lại với cả hai rủi ro.
  - Chú ý: (b) cũng bảo đảm rủi ro trung bình theo cụm: một ảnh ngẫu nhiên của cụm ngẫu nhiên có kỳ vọng loss = L_c, nên
    ảnh được chọn của cụm hiệu chuẩn và "một ảnh ngẫu nhiên của cụm mới" trao đổi được.
- Chi phí gán nhãn `cost_frames`: (a)/(c) = mọi khung của cụm rút (phải chú thích hết mới biết khung nào dương và loss của
  chúng); (b) = k khung. `cost_pos` = số ảnh dương dùng để hiệu chuẩn.
- Tái lập lần rút: cụm ~ `np.random.default_rng([2026, fold_idx, k, 11]).choice(sorted positive target clusters, k, replace=False)`
  lần lượt cho mỗi lần rút; khung cho (b) ~ `default_rng([2026, fold_idx, k, 12])`. Giống nhau giữa seed/IoU/α (common
  random numbers). 500 lần rút (`--draws`), 200 lần chia/bootstrap.

## Kiểm tra (assert trong script, stdout "SANITY ... OK")
1. L_c ∈ [0, 1] và không giảm theo λ (mọi run, cả hai IoU) — đạt.
2. λ̂ cấp cụm (vectorized `crc_idx`) == `metrics.crc_lambda` trên loss cụm (mọi ô pooled) và == vòng lặp brute-force
   (96 trường hợp: pooled + một bootstrap, α ∈ {0,10; 0,20}) — đạt.
3. C3 (toàn cụm đơn): CRC cấp cụm == CRC theo ảnh cho cả (a)/(b)/(c): tổng loss allclose, λ̂ trùng ngoại trừ **6 trường hợp
   hoà số thực** đúng tại biên α (tiêu chí = α tới 1e-9, khác thứ tự cộng); `risk_cl == risk_fr` — đạt.
4. Không cụm nào vừa ở cal vừa ở đích.

## Kết quả

### 1. In-domain: CRC cấp cụm xoá được phần vượt α (stdout)
| Chiến lược (cùng 200 lần chia cụm dương) | rủi ro được đo | ô không degenerate | max(trung bình − α) | ô > α + 0,02 |
|---|---|---|---|---|
| `in_domain_cl` | theo cụm (được bảo đảm) | 89/120 | **+0,0028** | 0 |
| `in_domain_cl` | theo khung | 89/120 | −0,0040 | 0 |
| `in_domain_img` | theo khung | 93/120 | **+0,0382** | 13 |
| `in_domain_img` | theo cụm | 93/120 | +0,1535 | 76 |

→ Giải thích của lượt đầu đúng: CRC theo ảnh với chia theo cụm vượt α (+0,038, lặp lại đúng con số lượt đầu; tệ nhất fold C3
α = 0,15 IoU 0,1 seed 1: 0,182), còn CRC cấp cụm đạt ≤ α + 0,003 ở mọi ô (120 ô = 6 fold × 2 seed × 2 IoU × 5 α; 31 ô
degenerate chủ yếu ở α = 0,05 và IoU 0,5 với α = 0,10). Ở in-domain, theo khung còn thấp hơn theo cụm (rủi ro theo khung
0,049–0,083 ở α = 0,10) — các video dài ở cal nguồn dễ hơn ảnh đơn. Tần suất vi phạm theo cụm 0,36–0,41 (α = 0,10).

### 2. Bảng chính (IoU 0,1). Ô: rủi ro cụm/khung (tần suất vi phạm cụm/khung). Pooled: ước lượng điểm (vi phạm qua 200 bootstrap × 2 seed). Cost = trung vị số khung phải chú thích cho (a)/(c); (b) tốn k khung.

α = 0,10:

| Đích | in-domain cl | pooled cl | a10 | b10 | c10 | cost10 | a20 | b20 | c20 | cost20 |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 0,095/0,069 (0,39/0,19) | 0,101/0,330 (0,49/0,99) | 0,061/0,199 (0,21/0,75) | 0,076/0,226 (0,28/0,79) | 0,082/0,251 (0,32/0,85) | 15 | 0,076/0,241 (0,26/0,83) | 0,086/0,254 (0,31/0,84) | 0,094/0,286 (0,40/0,85) | 31 |
| C2 | 0,090/0,083 (0,36/0,26) | 0,069/0,179 (0,01/1,00) | 0,069/0,158 (0,23/0,50) | 0,072/0,162 (0,24/0,54) | 0,073/0,165 (0,26/0,51) | 10 | 0,080/0,183 (0,29/0,78) | 0,084/0,188 (0,32/0,81) | 0,076/0,175 (0,28/0,66) | 20 |
| C3 | 0,090/0,049 (0,38/0,10) | 0,027/0,027 (0,00/0,00) | 0,069/0,069 (0,24/0,24) | = a | = a | 10 | 0,079/0,079 (0,29/0,29) | = a | = a | 20 |
| C4 | 0,094/0,069 (0,41/0,19) | 0,198/0,166 (0,92/0,92) | 0,038/0,047 (0,01/0,01) | 0,084/0,080 (0,26/0,21) | 0,110/0,100 (0,49/0,42) | 206 | 0,059/0,064 (0,13/0,12) | 0,085/0,079 (0,29/0,24) | 0,117/0,102 (0,53/0,47) | 686 |
| C5 | 0,092/0,061 (0,39/0,14) | 0,209/0,216 (1,00/1,00) | 0,084/0,087 (0,20/0,23) | 0,084/0,087 (0,21/0,23) | 0,084/0,087 (0,20/0,23) | 10 | 0,083/0,086 (0,25/0,27) | 0,083/0,086 (0,25/0,27) | 0,082/0,085 (0,24/0,26) | 20 |
| C6 | 0,095/0,063 (0,41/0,13) | 0,140/0,152 (1,00/1,00) | 0,075/0,083 (0,23/0,23) | 0,089/0,099 (0,30/0,30) | 0,107/0,121 (0,48/0,46) | 61 | 0,071/0,076 (0,21/0,21) | 0,079/0,084 (0,27/0,26) | 0,101/0,112 (0,48/0,50) | 121 |

α = 0,20:

| Đích | in-domain cl | pooled cl | a10 | b10 | c10 | cost10 | a20 | b20 | c20 | cost20 |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 0,194/0,145 (0,43/0,19) | 0,268/0,502 (0,99/1,00) | 0,150/0,360 (0,26/0,83) | 0,162/0,368 (0,31/0,82) | 0,189/0,409 (0,42/0,89) | 15 | 0,172/0,390 (0,32/0,87) | 0,179/0,396 (0,35/0,88) | 0,190/0,415 (0,46/0,86) | 31 |
| C2 | 0,194/0,183 (0,47/0,36) | 0,168/0,312 (0,05/1,00) | 0,155/0,286 (0,30/0,71) | 0,157/0,289 (0,31/0,72) | 0,152/0,284 (0,31/0,68) | 10 | 0,180/0,325 (0,38/0,91) | 0,183/0,328 (0,39/0,91) | 0,159/0,297 (0,33/0,77) | 20 |
| C3 | 0,193/0,105 (0,43/0,09) | 0,063/0,063 (0,00/0,00) | 0,161/0,161 (0,29/0,29) | = a | = a | 10 | 0,177/0,177 (0,35/0,35) | = a | = a | 20 |
| C4 | 0,197/0,159 (0,44/0,29) | 0,411/0,385 (1,00/1,00) | 0,138/0,118 (0,21/0,10) | 0,169/0,146 (0,33/0,24) | 0,230/0,201 (0,58/0,45) | 206 | 0,166/0,138 (0,32/0,17) | 0,177/0,152 (0,37/0,27) | 0,233/0,206 (0,59/0,49) | 686 |
| C5 | 0,195/0,138 (0,44/0,17) | 0,404/0,409 (1,00/1,00) | 0,160/0,166 (0,29/0,30) | 0,160/0,166 (0,29/0,30) | 0,156/0,162 (0,28/0,29) | 10 | 0,171/0,177 (0,34/0,36) | 0,171/0,178 (0,34/0,36) | 0,166/0,172 (0,31/0,33) | 20 |
| C6 | 0,196/0,155 (0,43/0,18) | 0,242/0,303 (0,92/1,00) | 0,164/0,193 (0,29/0,34) | 0,171/0,203 (0,33/0,37) | 0,186/0,227 (0,38/0,47) | 61 | 0,165/0,190 (0,28/0,36) | 0,168/0,195 (0,29/0,37) | 0,171/0,207 (0,35/0,47) | 121 |

Degenerate (α = 0,10, IoU 0,1): k = 10 nằm sát mức tối thiểu 9 cụm, nên (a) degenerate 0,37 / 0,40 / 0,27 / **0,91** / 0,56 /
0,49 ở C1…C6 (C4 a10 "an toàn" chủ yếu nhờ thất bại: λ̂ = 0,001); ở k = 20: 0,12 / 0,07 / 0,05 / 0,04 / 0,41 / 0,33; k = 40:
≤ 0,20. k = 5 degenerate 100 % ở α ≤ 0,15. Vì thế k = 20 là so sánh có nghĩa ở α = 0,10.

### 3. Đọc
- **Pooled cấp cụm vẫn hỏng khi đổi bệnh viện** ở C4, C5, C6 (rủi ro cụm 0,14–0,21 ở α = 0,10, vi phạm bootstrap 0,92–1,00).
  Ở C1 và C2 rủi ro *theo cụm* gần/dưới α (0,101; 0,069) nhưng rủi ro theo khung vẫn 0,330 / 0,179 — tức "sửa" ở C1/C2 chỉ
  là đổi đại lượng đo: các ảnh đơn (đa số cụm) dễ, video dài khó. Dịch chuyển miền không được CRC cấp cụm giải quyết; nó
  chỉ sửa vi phạm trao đổi được *trong* miền.
- **C1/C2 few-shot**: theo cụm thì đạt (C1 a20 0,076, C2 a20 0,080 ở α = 0,10; vi phạm 0,26–0,29 < mức in-domain ~0,4),
  theo khung thì không (C1 0,24, C2 0,18; vi phạm khung 0,78–0,83). Như vậy CRC cấp cụm *không* sửa được C1/C2 theo nghĩa
  của lượt đầu (rủi ro theo khung); nó sửa theo nghĩa nó bảo đảm (rủi ro theo video). Không có lựa chọn nào trong ba thiết
  kế đưa rủi ro theo khung ở C1 về α — điều đó cần nhiều nhãn ở chính seq11 hoặc trọng số theo độ dài video.
- **Ngân sách đếm bằng video**: với cùng k video, (a) và (b) có rủi ro cụm ≤ α ở mọi fold và mọi k (bảng + hình), còn (c)
  (CRC theo ảnh trên mọi khung của k video) vượt α theo cụm ở C4 (0,110–0,118) và C6 (0,097–0,107), vi phạm ~0,5. Few-shot
  theo ngân sách ảnh (`fs_frames`) cũng vượt ở C4 (0,117–0,134) và C6 (0,111–0,121) theo cụm.
- **Chi phí** (trung vị khung phải chú thích, α-độc lập; `cost_frames_median`): (b) = k khung. (a)/(c) với k = 20: C1 31,
  C2 20, C3 20, **C4 686**, C5 20, **C6 121**; k = 40: C2 157,5, C4 792 (k = 24), C6 244. Ngân sách ảnh 100 dương
  (`fs_frames`) tốn 310 / 273 / 100 / 525 / 381,5 / 565 khung và chỉ cho trung vị 57 / 37 / 100 / 9 / 99 / 23 cụm.
  → **Một khung mỗi video (b) đạt gần cùng rủi ro cụm như (a) với chi phí bằng k khung**: ở C4 k = 20, 20 khung thay vì 686
  (×34); C6 20 thay vì 121. Ở nơi gần như toàn ảnh đơn (C2, C3, C5) ba thiết kế trùng nhau.
- (a) bảo thủ hơn (b) ở C4/C6 (C4 a20 0,059 vs b20 0,085): trung bình trên nhiều khung làm L_c nhỏ phương sai nhưng bảo đảm
  giống nhau; (b) "tiêu" hết α hơn → ít cảnh báo giả hơn ở k nhỏ (C4 FA/khung âm α = 0,10: a10 9,26 vs b10 5,53; k = 20 hai
  bên ~4,3). (c) có FA thấp nhất ở C4/C6 (2,1 / 1,7 ở k = 20) vì nó vượt α.
- **Worst-case cấp cụm vô dụng ở dạng thô**: cal nguồn C2 chỉ có **1** cụm dương (`C2_seq12`) và C4 có 4 → λ̂_C2 luôn
  degenerate với α < 0,5, nên `worst_cl` degenerate 100 % ở α ≤ 0,20 cho mọi fold có C2 là nguồn. `worst_cl_feas` (bỏ trung
  tâm không đủ cỡ): α = 0,10 → không degenerate ở điểm ước lượng, rủi ro cụm 0,029 / 0,023 / 0,017 / 0,038 / 0,104 / 0,103
  (C5, C6 vẫn > α), FA rất cao (C1 9,6, C4 4,7, C5 3,3 box/khung âm); α = 0,20 → C4 (4 cụm) được tính là khả thi nhưng
  degenerate → 83 % fold degenerate. Kết luận: chiến lược theo từng trung tâm cần ≥ 1/α − 1 *video* mỗi trung tâm nguồn, mà
  PolypGen không có.
- Micro miss (α = 0,10, pooled_cl): 0,338 / 0,194 / 0,075 / 0,179 / 0,251 / 0,174 (C1…C6). FA/khung âm pooled_cl: C1 0,87,
  C4 0,62, C5 0,44, C6 0,11.
- IoU 0,5 (α = 0,20, k = 20): rủi ro cụm (a) 0,165–0,181, (b) 0,169–0,198, (c) 0,164–0,283 (C4 0,283); degenerate (a)
  C4 0,51, C5 0,37. In-domain cl ≤ α. Cùng hướng với IoU 0,1 nhưng degenerate nhiều hơn.

## Cái gì được, cái gì không
- Được: (1) bảo đảm in-domain chính xác lại khi đơn vị là cụm (+0,038 → +0,003); (2) với ngân sách đếm bằng video, (a) và
  (b) đạt α theo cụm ở cả 6 đích từ k ≈ 10–20 video; (3) (b) rẻ hơn (a)/(c) tới ×34 khung ở trung tâm nhiều video dài.
- Không được: (1) pooled vẫn hỏng ở C4–C6 — dịch chuyển miền vẫn là vấn đề; (2) rủi ro theo khung ở C1/C2 vẫn > α với mọi
  thiết kế — CRC cấp cụm đổi câu hỏi chứ không sửa nó; (3) worst-case theo trung tâm không khả thi ở cấp cụm.

## Hạn chế
- Mới 2 seed (0, 1). Seed 2 về → chạy lại cùng lệnh (file bị ghi đè), rồi chạy `scripts/fig_cluster_budget.py`. Error bar =
  khoảng min–max 2 seed. YOLO11m: `--root outputs/kaggle_m`.
- Cụm chỉ là ước lượng video (`docs/clusters.md`): ảnh đơn C2/C3/C5/C6 có thể cùng video mà không nối được → số cụm là cận
  trên số đơn vị độc lập; bảo đảm cấp cụm vẫn lạc quan ở mức đó.
- Rủi ro theo cụm cho mỗi video trọng số bằng nhau; nếu lâm sàng quan tâm tỷ lệ bỏ sót theo khung (hoặc theo thời gian
  nội soi) thì đây không phải đại lượng đúng.
- Chặn k ở (số cụm dương − 10): C4 chỉ 34 cụm dương, k = 40 → 24 và test còn 10 cụm (nhiễu lớn).
- Tần suất vi phạm không được bảo đảm (CRC bảo đảm kỳ vọng).

## Có phải đóng góp phương pháp đứng được không?
Có, nhưng khiêm tốn và phải nói đúng. CRC cấp cụm là áp dụng thẳng CRC (Angelopoulos et al., 2024) với đơn vị = video và
loss = trung bình loss khung trong video (giống "group/cluster conformal" đã có trong văn liệu cho dữ liệu phân cấp) — không
phải thuật toán mới. Điểm đứng được là **thực nghiệm + thiết kế ngân sách**: (i) chỉ ra bằng số rằng CRC theo khung vi phạm
bảo đảm trong video nội soi (+0,038 in-domain) và cấp cụm sửa được (+0,003); (ii) "một khung mỗi video" giữ cùng bảo đảm với
chi phí k khung, rẻ hơn tới ×34 so với chú thích cả video — khuyến nghị thực hành cụ thể cho hiệu chuẩn lại tại bệnh viện mới.
Phát biểu bảo đảm chính xác: *Nếu k video dương hiệu chuẩn và một video dương mới của bệnh viện đích là trao đổi được (ví dụ
rút i.i.d. từ phân phối video của bệnh viện đó), và L_c(λ) ∈ [0, 1] không giảm theo λ, thì với λ̂ như trên,
E[L_{new}(λ̂)] ≤ α, trong đó L_new là tỷ lệ polyp bị bỏ sót trung bình trên các khung dương của video mới; kỳ vọng lấy trên
cả tập hiệu chuẩn và video mới. Thiết kế (b) cho cùng bảo đảm với L_new thay bằng loss của một khung dương ngẫu nhiên đều của
video mới (cùng kỳ vọng).* Không có bảo đảm cho rủi ro theo khung, cho từng lần rút, hay khi đổi bệnh viện mà không hiệu
chuẩn lại tại đích.

## Hình
`results/figures/fig_cluster_budget.{pdf,eps,png}` (183 × 90 mm). Draft caption: *Few-shot recalibration at the held-out
hospital with the labelling budget counted in videos (IoU ≥ 0.1). Mean cluster-averaged target miss rate minus α versus the
number of labelled positive target videos, for α = 0.10 (top) and α = 0.20 (bottom) and each held-out centre (a–f).
Blue, cluster-level CRC on all frames of k videos; orange, image-level CRC on one random positive frame per video; green
dashed, image-level CRC on 20/50/100 positive frames placed at the median number of videos those frames came from. Values
below zero meet the guarantee; points at small k are partly driven by degenerate (all-box) thresholds. Means over 500 draws
and two training seeds; error bars show the range across seeds.*
