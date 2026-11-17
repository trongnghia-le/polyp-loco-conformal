# 2026-11-09 — Hiệu chuẩn nguồn cân bằng theo trung tâm (R6a–c), trả lời phản biện vòng 2, M1

Script: `scripts/revision2_balanced.py` (chạy: `.venv/Scripts/python.exe -I scripts/revision2_balanced.py`, khoảng 3 phút CPU).
Chỉ dùng dự đoán có sẵn của YOLO11s (`outputs/kaggle`, 18 run). Không huấn luyện, không dữ liệu mới, không sửa script cũ.
Script import `conformal.load_run`, `crc_idx`, `G`, `ALPHAS`, `IOUS`, `metrics.crc_lambda` và `revision_analyses.interval`/`B`.
Loss và quy tắc CRC không viết lại.

| File | Nội dung |
|---|---|
| `outputs/eval/revision2_R6a_weighted.csv` | R6a: CRC có trọng số cân bằng trung tâm. `level` ∈ {frame, cluster}; `variant` ∈ {equal (= pooled/pooled_cl đã công bố), balanced_conservative, balanced_plugin}. Theo fold × seed (0/1/2/all) × IoU × α. Có risk_fr, risk_cl, fa_neg, λ̂, degenerate, n_eff, p_test và khoảng bootstrap cụm đích (lo/hi/boot_frac_above, cho cả fr và cl). Với seed = all thêm `seeds_above_*` |
| `outputs/eval/revision2_R6b_stratified.csv` | R6b: mỗi trung tâm nguồn lấy đúng m đơn vị dương, 200 lần rút. Có m, n_cal, feasible (n ≥ 1/α − 1), degenerate_frac, risk_fr/risk_cl (mean, q05, q95), viol_fr/viol_cl (tỷ lệ lần rút × seed > α), fa_neg |
| `outputs/eval/revision2_R6c_loso_cluster.csv` | R6c: pooled_cl bỏ lần lượt từng trung tâm nguồn (`excluded`). Có risk_cl, risk_fr, fa_neg, degenerate, seeds_above_* |

Mặc định trong ghi chú này là IoU ≥ 0,1, α = 0,10, seed = "all". Các α khác (0,05/0,15/0,20/0,30) và IoU 0,5 có trong CSV.
Ô "điểm [2,5 %; 97,5 %] P" nghĩa là: điểm = trung bình qua 3 seed; khoảng = bootstrap cụm đích phân cấp, giống hệt R1 (B = 1 000, cùng mẫu bootstrap, mỗi mẫu rút 1 seed); P = tỷ lệ mẫu bootstrap > α.

## Số đơn vị dương trong tập cal nguồn (giống nhau ở mọi seed)
- Khung dương: C1 78, C2 252, C3 91, C4 76, C5 41, C6 81.
- Cụm dương: C1 57, **C2 1**, C3 91, **C4 4**, C5 40, C6 27.

Vì vậy n_min là:
- cấp khung: 41 (C5), riêng fold C5 là 76 (C4);
- cấp cụm: **1** (C2) ở mọi fold trừ fold C2, nơi n_min = 4 (C4).

## Phương pháp

### R6a — CRC có trọng số (Angelopoulos et al., *Conformal Risk Control*, mục 4.1, mở rộng Tibshirani et al. 2019)

**Thiết lập.**
- Đơn vị u là một khung dương của tập cal nguồn (cấp khung, loss L_i như `conformal.py`), hoặc một cụm dương (cấp cụm, L_c = trung bình L_i trên các khung dương của cụm, như `conformal_cluster.py`).
- Đơn vị u thuộc trung tâm k(u). Trung tâm k có n_k đơn vị; có K = 5 trung tâm nguồn.
- Phân phối mục tiêu Q = (1/K) Σ_k P_k: hỗn hợp đều các trung tâm nguồn, mỗi trung tâm nặng như nhau.
- Tập cal được coi là mẫu từ P = Σ_k (n_k/n) P_k.
- Q và P chỉ khác nhau ở nhãn trung tâm (dịch chuyển hiệp biến theo nhãn), với tỷ số hợp lý đã biết w(u) = dQ/dP = (1/K)/(n_k/n) ∝ 1/n_k.

**Quy tắc** (đổi chiều λ vì loss của ta không giảm theo λ, nên lấy sup thay cho inf):

λ̂ = sup{λ : Σ_u p_u L_u(λ) + p_test · B ≤ α}, với B = 1, p_u = w_u/(Σw + w_test), p_test = w_test/(Σw + w_test).

**Trọng số của điểm test chưa biết, nên chọn w_test = max_u w_u = trọng số của trung tâm nhỏ nhất.** Lựa chọn này bảo thủ vì:
- Vế trái bằng (Σ w_u L_u + w_test)/(Σw + w_test). Đạo hàm theo w_test là (Σw − Σ w_u L_u)/(Σw + w_test)² ≥ 0, vì L ≤ 1. Vậy vế trái tăng theo w_test.
- Do đó λ̂ với w_test = max nhỏ hơn hoặc bằng λ̂ với trọng số của mọi trung tâm test có thể có.
- L không giảm theo λ, nên đảm bảo của bản theo từng trung tâm truyền sang bản bảo thủ.

**Dạng đóng.**
- Chuẩn hoá w'_u = w_u / max w = n_min/n_k. Quy tắc trở thành (Σ w'_u L_u(λ) + 1)/(Σ w' + 1) ≤ α, với Σ w' = K·n_min.
- Đây đúng là `conformal.crc_idx(w'@L, Σw')`, nên tái dùng nguyên hàm này.
- Viết cách khác: (K·n_min · R̂_bal(λ) + 1)/(K·n_min + 1) ≤ α, trong đó R̂_bal = (1/K) Σ_k (trung bình L trong trung tâm k).
- Tức là: **CRC thường trên rủi ro thực nghiệm cân bằng theo trung tâm, với cỡ mẫu hiệu dụng K·n_min.**
- Điều kiện khả thi: 1/(K·n_min + 1) ≤ α.
  - Cấp khung: K·n_min = 205 (fold C5: 380), luôn khả thi.
  - Cấp cụm: K·n_min = 5, nên p_test = 1/6 = 0,167. **Không thể chứng nhận α < 0,167 ở 5/6 fold** (degenerate ở α = 0,05/0,10/0,15, nghĩa là giữ mọi box).
  - Fold C2 ở cấp cụm: K·n_min = 20, p_test = 1/21, khả thi.

**Plug-in (độ nhạy).** Bỏ số hạng p_test·B: Σ w_u L_u / Σ w_u ≤ α, tức R̂_bal ≤ α. Bản này không có hiệu chỉnh mẫu hữu hạn, nên không có đảm bảo.

**Giới hạn của đảm bảo** (phải nêu rõ trong bài):
1. Định lý giả định tập cal là mẫu iid từ P, tức thành phần trung tâm là ngẫu nhiên. Ở đây n_k cố định do thiết kế chia. Vì vậy quy tắc là cách ước lượng tự nhiên cho Q, còn đảm bảo chính xác chỉ thuộc về mô hình lấy mẫu ngẫu nhiên.
2. Ở cấp khung, các khung trong cùng video không hoán đổi được (giống pooled đã công bố).
3. Q là hỗn hợp nguồn, không phải trung tâm bị giữ lại. Ở trung tâm mới, CRC không đảm bảo gì. Câu hỏi ở đây là câu hỏi thực nghiệm: quy tắc cân bằng có chuyển sang được hay không.

### R6b — Lấy mẫu con phân tầng, cùng số lượng mỗi trung tâm
- Mỗi trung tâm nguồn: rút m = min_k n_k đơn vị dương, không hoàn lại. 200 lần rút, độc lập với seed, sinh bởi `default_rng([2026, fold, 66, level])`.
- Chạy CRC thường trên K·m đơn vị.
- Cấp khung: m = 41 (fold C5: 76), n = 205/380, khả thi ở mọi α.
- Cấp cụm: m = 1 (fold C2: m = 4), n = 5 (20). Chỉ khả thi khi α ≥ 1/6 (fold C2: α ≥ 1/21). `feasible` trong CSV.

### R6c — Bỏ từng trung tâm nguồn, cấp cụm
- Chạy pooled_cl sau khi bỏ toàn bộ cụm cal của một trung tâm nguồn.
- Tái lập số "không C3" của phản biện.

## Kiểm tra (assert trong script, in ra stdout)
- **360 ô**: CRC có trọng số với trọng số bằng nhau == `metrics.crc_lambda` == pooled (cấp khung, λ̂ và risk so với `conformal_summary.csv`) và == pooled_cl (cấp cụm, λ̂, risk_cl, risk_fr so với `conformal_cluster_summary.csv`). Sai khác < 1e-12.
- Khoảng bootstrap của biến thể equal trùng với `revision_R1_target_bootstrap.csv` (pooled, pooled_cl; cả hai đại lượng; mọi seed, IoU, α; < 1e-12).
- Tổng trọng số chuẩn hoá Σw' = K·n_min (mọi ô).
- R6c bỏ C3 == số của phản biện (làm tròn 3 chữ số): C1 0,054; C2 0,054; C4 0,095; C5 0,181; C6 0,107.
  Script `cl_noC3.py` của phản biện dùng đúng `load_run`/`crc_idx` và cùng định nghĩa loss cụm. Cột "all" của nó trùng pooled_cl.

## Kết quả, α = 0,10, IoU ≥ 0,1

### Rủi ro trung bình theo khung ở trung tâm đích (CRC cấp khung)
Cột phân tầng: trung bình qua 200 lần rút × 3 seed, (q05–q95 qua lần rút), tỷ lệ lần rút > α. Biến thể này không có bootstrap đích.

| Đích | pooled đã công bố | cân bằng bảo thủ | cân bằng plug-in | phân tầng m = 41 (C5: 76) |
|---|---|---|---|---|
| C1 | 0,468 [0,126; 0,706] 1,00 | 0,373 [0,068; 0,668] 0,83 | 0,382 [0,069; 0,671] 0,86 | 0,375 (0,262–0,437) 1,00 |
| C2 | 0,198 [0,061; 0,289] 0,93 | 0,180 [0,058; 0,263] 0,90 | 0,186 [0,059; 0,276] 0,92 | 0,180 (0,149–0,209) 1,00 |
| C3 | 0,100 [0,062; 0,157] 0,38 | 0,054 [0,023; 0,100] 0,03 | 0,058 [0,028; 0,104] 0,05 | 0,056 (0,034–0,090) 0,00 |
| C4 | 0,288 [0,164; 0,425] 1,00 | 0,159 [0,070; 0,258] 0,86 | 0,171 [0,073; 0,277] 0,88 | 0,167 (0,083–0,247) 0,91 |
| C5 | 0,375 [0,305; 0,446] 1,00 | **0,312 [0,243; 0,378] 1,00** | 0,318 [0,252; 0,385] 1,00 | 0,312 (0,275–0,358) 1,00 |
| C6 | 0,275 [0,144; 0,432] 1,00 | **0,192 [0,103; 0,300] 0,98** | 0,201 [0,109; 0,310] 0,99 | 0,189 (0,140–0,223) 1,00 |

- Ở cấp khung, ba biến thể cân bằng gần như trùng nhau. Điều này đúng như dự kiến: cùng R̂_bal, và p_test = 1/206 là không đáng kể.
- Mọi seed (3/3) đều > α ở C1, C2, C4, C5, C6. Ở C3 không seed nào > α.

### Rủi ro trung bình theo video ở trung tâm đích
| Đích | pooled_cl đã công bố | cụm, cân bằng bảo thủ | cụm, cân bằng plug-in | cụm, phân tầng | khung, cân bằng bảo thủ (đánh giá theo video) |
|---|---|---|---|---|---|
| C1 | 0,113 [0,064; 0,176] 0,66 | không chứng nhận được (sàn 0,021) | 0,140 [0,092; 0,192] 0,95 | 100 % degenerate (m = 1) | 0,129 [0,073; 0,189] 0,82 |
| C2 | 0,074 [0,045; 0,109] 0,07 | 0,030 [0,011; 0,053] 0,00 (n_eff 20) | 0,070 [0,042; 0,098] 0,02 | 0,046 (m = 4; 27 % degenerate; 0,075 lần rút > α) | 0,066 [0,040; 0,097] 0,01 |
| C3 | 0,037 [0,017; 0,070] 0,00 | không chứng nhận được (0,015) | 0,049 [0,026; 0,083] 0,00 | 100 % degenerate | 0,054 [0,023; 0,100] 0,03 |
| C4 | 0,174 [0,090; 0,285] 0,95 | không chứng nhận được (0,030) | **0,214 [0,126; 0,333] 1,00** | 100 % degenerate | 0,181 [0,083; 0,300] 0,92 |
| C5 | 0,218 [0,154; 0,283] 1,00 | không chứng nhận được (0,054) | **0,262 [0,194; 0,347] 1,00** | 100 % degenerate | **0,306 [0,239; 0,371] 1,00** |
| C6 | 0,144 [0,078; 0,221] 0,89 | không chứng nhận được (0,040) | **0,178 [0,102; 0,258] 0,98** | 100 % degenerate | **0,182 [0,106; 0,262] 0,99** |

"Không chứng nhận được" là trường hợp degenerate: λ̂ = 0,001, giữ mọi box, nên rủi ro bằng sàn. Đây không phải là "đạt". Cụm, cân bằng plug-in, đánh giá theo khung: C1 0,394; C2 0,180; C3 0,049; C4 0,179; C5 0,268; C6 0,191.

### Cảnh báo giả (box / khung không polyp ở đích; C1 / C4 / C5 / C6)
- pooled khung: 0,12 / 0,39 / 0,15 / 0,02.
- khung cân bằng bảo thủ: 0,52 / 0,90 / 0,23 / 0,06; plug-in: 0,47 / 0,81 / 0,22 / 0,05.
- phân tầng khung: 0,54 / 0,96 / 0,23 / 0,07.
- pooled_cl: 0,62 / 0,92 / 0,48 / 0,12; cụm cân bằng plug-in: 0,38 / 0,78 / 0,33 / 0,06.
- cụm cân bằng bảo thủ (degenerate) và phân tầng cụm: 15,0 / 10,4 / 7,6 / 3,9.

Tức là cân bằng ở cấp khung làm ngưỡng chặt hơn (λ̂ ở C1 0,308 → 0,102) và FA tăng khoảng 2–4 lần, nhưng miss vẫn > α.

### R6c — pooled_cl bỏ từng trung tâm nguồn (risk_cl / risk_fr; tỷ lệ seed > α theo video)
| Bỏ | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| không bỏ (pooled_cl) | 0,113 / 0,358 | 0,074 / 0,188 | 0,037 | 0,174 / 0,148 | 0,218 / 0,225 | 0,144 / 0,157 |
| C1 | — | 0,070 / 0,183 (0,00) | 0,032 | 0,180 / 0,160 (1,00) | 0,249 / 0,255 (1,00) | 0,161 / 0,175 (1,00) |
| C2 | 0,113 / 0,358 (0,67) | — | 0,035 | 0,171 / 0,146 (1,00) | 0,218 / 0,225 (1,00) | 0,144 / 0,157 (1,00) |
| C3 | **0,054** / 0,218 (0,00) | **0,054** / 0,150 (0,00) | — | **0,095** / 0,087 (0,33) | **0,181** / 0,188 (1,00) | **0,107** / 0,113 (0,67) |
| C4 | 0,113 / 0,357 (0,67) | 0,074 / 0,188 (0,00) | 0,037 | — | 0,218 / 0,225 (1,00) | 0,144 / 0,157 (1,00) |
| C5 | 0,166 / 0,429 (1,00) | 0,088 / 0,211 (0,33) | 0,043 | 0,231 / 0,197 (1,00) | — | 0,190 / 0,200 (1,00) |
| C6 | 0,124 / 0,364 (0,67) | 0,075 / 0,189 (0,00) | 0,035 | 0,176 / 0,152 (1,00) | 0,242 / 0,248 (1,00) | — |

FA khi bỏ C3 (C1/C4/C5/C6): 3,10 / 2,47 / 0,79 / 0,49, so với pooled_cl 0,62 / 0,92 / 0,48 / 0,12.

## Đọc

1. **Phản biện tái lập đúng** (bỏ C3). Tuy nhiên bỏ C3 là trường hợp đặc biệt.
   - Trong năm phép bỏ một trung tâm, chỉ bỏ C3 hạ được rủi ro theo video. Bỏ C1, C2, C4, C6 gần như không đổi; bỏ C5 (trung tâm nguồn khó nhất) làm tăng.
   - C3 là trung tâm dễ nhất (sàn 1,5 %). Bỏ hẳn C3 làm ngưỡng chặt lại, với cái giá là FA tăng 1,5–5 lần.
2. **Cân bằng theo trung tâm (đúng đề xuất M1a) không cứu được kết quả theo video, mà làm nó tệ hơn.**
   - Ở cấp cụm, cân bằng hạ C3 từ 41 % xuống 20 % trọng số, nhưng nâng *một video C2 duy nhất* và *4 video C4* lên 20 % mỗi trung tâm. Video C2 cũng dễ.
   - Kết quả là λ̂ lỏng hơn (C1: 0,060 → 0,102) và rủi ro theo video ở C1/C4/C5/C6 tăng: 0,113 → 0,140; 0,174 → 0,214; 0,218 → 0,262; 0,144 → 0,178.
   - Vậy câu "thất bại theo video ở C1/C4/C6 phần lớn là do trọng số C3" chỉ đúng với một cách *bỏ* C3, không đúng với cách *cân bằng* các trung tâm.
3. **Ở cấp cụm, bản cân bằng có đảm bảo mẫu hữu hạn không thể chứng nhận α = 0,10.** Lý do: C2 chỉ có 1 video dương trong cal, nên p_test = 1/6. Đây là hạn chế của thiết kế chia, không phải phát hiện.
   - Ở α = 0,20 thì bản bảo thủ cụm khả thi, và rủi ro theo video ≤ α ở cả 6 trung tâm: C1 0,035; C2 0,123; C3 0,017; C4 0,049; C5 0,105; C6 0,072.
   - Nhưng đó là vì số hạng 1/6 ăn gần hết ngân sách, buộc R̂_bal ≤ 0,04. λ̂ chỉ còn 0,002–0,005 (C2 0,150), và FA là 6,8 / 5,0 / 2,0 / 1,2 box mỗi khung âm (C1/C4/C5/C6), so với 0,10 / 0,21 / 0,08 / 0,01 của pooled_cl.
   - Phải báo kết quả này. Đây là "đạt bằng cách gần như không lọc box".
4. **Ở cấp khung, cân bằng giảm độ lớn, nhưng chiều không đổi.**
   - Phần vượt α giảm ở C1 (0,368 → 0,273), C4 (0,188 → 0,059), C5 (0,275 → 0,212), C6 (0,175 → 0,092). C2 gần như không đổi (0,098 → 0,080).
   - Kết quả gần giống bản "1 khung/video" của R3 (0,15–0,36). Hai cách sửa trọng số độc lập cho cùng một bức tranh.
   - Phân tầng R6b ở cấp khung cho đúng số của bản cân bằng bảo thủ (cùng cỡ mẫu hiệu dụng 205), nên kiểm chứng chéo được.
5. **IoU 0,5** (CSV). Khung cân bằng bảo thủ theo khung: C1 0,332 (P 0,74); C2 0,152 (0,83); C3 0,045; C4 0,160 (0,94); C5 0,302 (1,00); C6 0,203 (0,97). Ở tiêu chí này sàn theo ảnh đã > 0,10 ở C1, C2, C4, C5, C6 (R5), nên "thất bại" chủ yếu là do sàn. Ở cấp cụm, bản bảo thủ degenerate ở mọi fold.
6. **α = 0,20, IoU 0,1.** Khung cân bằng bảo thủ theo khung: C1 0,523 (0,99); C2 0,307 (0,90); C3 0,147 (0,04); C4 0,369 (1,00); C5 0,466 (1,00); C6 0,353 (0,99). Cụm cân bằng plug-in theo video: C1 0,270 (0,96); C2 0,167; C3 0,118; C4 0,418; C5 0,435; C6 0,274 (0,87).

## Phán quyết theo trung tâm đích (α = 0,10, IoU ≥ 0,1, quy tắc cân bằng theo trung tâm)
Quy ước:
- *vẫn hỏng*: điểm > α và cận 2,5 % > α;
- *sát biên / có khả năng*: điểm > α nhưng khoảng chứa α;
- *hết hỏng*: điểm ≤ α.

| Đích | theo khung (khung cân bằng bảo thủ) | theo video (cụm cân bằng plug-in; bảo thủ thì không chứng nhận được) | so với pooled đã công bố |
|---|---|---|---|
| C1 | sát biên / có khả năng (0,373; cận dưới 0,068; P 0,83) | sát biên (0,140; cận dưới 0,092; P 0,95) | theo khung: từ "chắc chắn" xuống "có khả năng"; theo video: mạnh hơn (0,66 → 0,95) |
| C2 | sát biên (0,180; P 0,90) | hết hỏng (0,070) | như cũ |
| C3 | hết hỏng (0,054) | hết hỏng (0,049) | như cũ (theo khung trước là sát α) |
| C4 | sát biên (0,159; cận dưới 0,070; P 0,86) | **vẫn hỏng** (0,214; cận dưới 0,126) | theo khung: yếu đi; theo video: mạnh hơn (0,95 → 1,00) |
| C5 | **vẫn hỏng** (0,312 [0,243; 0,378]) | **vẫn hỏng** (0,262 [0,194; 0,347]) | như cũ, chắc chắn ở mọi biến thể |
| C6 | **vẫn hỏng**, sát (0,192; cận dưới 0,103) | **vẫn hỏng**, sát (0,178; cận dưới 0,102) | theo video: mạnh hơn (0,89 → 0,98) |

Thất bại còn lại sau cân bằng, có thể chống lại được, là: C5 và C6 ở cả hai đại lượng, cộng C4 theo video. Ở C1 (cả hai đại lượng), C2 và C4 (theo khung), điểm vẫn > α nhưng khoảng chứa α. C3 không hỏng. C2 theo video không hỏng.

## Hạn chế của các phân tích này
- Khoảng là bootstrap cụm đích với λ̂ cố định (cùng ước lượng như R1). Nhận xét M3 vẫn áp dụng: biến thiên của tập cal nguồn chưa được lồng vào. R6b q05–q95 cho thấy phần biến thiên đó (C4 theo khung 0,083–0,247).
- Bản cụm có đảm bảo (bảo thủ) không trả lời được ở α ≤ 0,15. Bản plug-in thì không có hiệu chỉnh mẫu hữu hạn, và trọng số của C2 đặt lên đúng một video.
- Đảm bảo của CRC có trọng số giả định thành phần trung tâm là ngẫu nhiên. Ở đây thành phần là cố định.
- Không làm hình: hiện có 8/8 display item. Nếu cần, có thể thêm một panel vào `fig_calibration_composition` hoặc đưa vào Supplement (chưa làm).

Draft caption (nếu đưa vào Supplement, dạng bảng): *Pooled-source CRC at the held-out centre with a centre-balanced source calibration (IoU ≥ 0.1, α = 0.10). Equal-weight (as published), centre-balanced weighted CRC (conservative test weight; plug-in without the finite-sample term) and equal-count stratified subsampling (200 draws), at frame and video level; realised frame- and video-averaged miss rate, mean over three seeds with target-video bootstrap 95% percentile intervals (B = 1,000, seed drawn per resample).*
