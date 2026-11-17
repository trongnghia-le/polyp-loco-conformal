# 2026-11-09 — Phân tích lại cho phản biện vòng 1 (R1–R5), không huấn luyện thêm

Script: `scripts/revision_analyses.py` (chạy: `.venv/Scripts/python.exe -I scripts/revision_analyses.py`, ~20 phút CPU).
Chỉ dùng kết quả có sẵn của YOLO11s (`outputs/kaggle`, 18 run) và các file `outputs/eval/*`. Import `conformal.py`,
`conformal_cluster.py`, `metrics.py` (không sửa); hàm `match`/`evaluate` của `eval_detection.py` được exec riêng (file đó
không có main guard nên không import trực tiếp). Đầu ra:

| File | Nội dung |
|---|---|
| `outputs/eval/revision_R1_target_bootstrap.csv` | R1: điểm, khoảng 2,5–97,5 %, tỷ lệ bootstrap > α; fold × seed × IoU × α × chiến lược × rủi ro (risk_fr = trung bình theo khung, risk_cl = theo video) |
| `outputs/eval/revision_R2_indomain_spread.csv` | R2: phân phối rủi ro in-domain qua 600 lần chia (200 × 3 seed) vs một lần triển khai pooled |
| `outputs/eval/revision_R3_calibration_composition.csv` | R3: pooled CRC với ≤ m khung dương/video và bỏ từng trung tâm nguồn |
| `outputs/eval/revision_R4_degenerate_fa.csv` | R4: degenerate, rủi ro trên lần rút không degenerate vs mọi lần rút, cảnh báo giả |
| `outputs/eval/revision_R5_detection_floor.csv` | R5: AP50 trên khung dương vs mọi khung; sàn bỏ sót theo ảnh / theo polyp / theo video |

Hình: `scripts/fig_calibration_composition.py` → `results/figures/fig_calibration_composition.{pdf,eps,png}` (89 mm).

## Kiểm tra (assert trong script, stdout)
- **1 332 lần tái lập đúng số cũ** (sai khác < 1e-12) *trước* mọi biến thể: pooled (rủi ro + λ̂) và worst_case so với
  `conformal_summary.csv`; pooled_cl, fs_a/fs_b k = 20 (rủi ro theo video và theo khung, cùng 500 lần rút) so với
  `conformal_cluster_summary.csv`; R3 m = all, không bỏ trung tâm == pooled; AP50 == `detection_summary.csv`;
  1 − max_recall == tỷ lệ bỏ sót polyp ở λ = 0,001 (`max_recall.csv`, sai khác < 1e-9); fs_frames == few_shot.
- R1: 2 400 khoảng, **mọi khoảng chứa ước lượng điểm**. Không lần rút nào có tập test rỗng.

Mặc định dưới đây: IoU ≥ 0,1, α = 0,10, seed = "all". Các số khác (IoU 0,5, α khác, từng seed) có trong CSV.

## R1 — Khoảng tin cậy cho rủi ro ở trung tâm đích (M4, M5.1)
**Cách làm.** B = 1 000 mẫu bootstrap: rút lại *cụm dương của trung tâm đích* có hoàn lại (cùng mẫu cho cả 3 seed).
- Pooled / worst_case / pooled_cl: λ̂ cố định (từ cal nguồn), tính lại rủi ro trên mẫu bootstrap.
- Hiệu chuẩn lại k = 20 video (fs_a, fs_b): chạy lại *toàn bộ* thủ tục trong mỗi mẫu bootstrap. Cụ thể: 100 lần rút 20 đơn vị → λ̂; test = các đơn vị mà video không nằm trong lần rút; lấy trung bình qua 100 lần rút.
- Dòng seed "all" là bootstrap phân cấp: mỗi mẫu bootstrap lấy ngẫu nhiên 1 trong 3 seed, nên khoảng phản ánh cả biến thiên huấn luyện (chỉ 3 seed) lẫn mẫu đích.

Ô: điểm [2,5 %; 97,5 %] P(bootstrap > α).

| | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| pooled, theo khung | 0,468 [0,126; 0,706] 1,00 | 0,198 [0,061; 0,289] 0,93 | 0,100 [0,062; 0,157] 0,38 | 0,288 [0,164; 0,425] 1,00 | 0,375 [0,305; 0,446] 1,00 | 0,275 [0,144; 0,432] 1,00 |
| worst_case, theo khung | 0,167 [0,010; 0,372] 0,43 | 0,080 [0,024; 0,125] 0,18 | 0,020 [0,009; 0,037] 0,00 | 0,051 [0,017; 0,112] 0,06 | 0,137 [0,076; 0,221] 0,81 | 0,087 [0,023; 0,178] 0,33 |
| pooled_cl, theo video | 0,113 [0,064; 0,176] 0,66 | 0,074 [0,045; 0,109] 0,07 | 0,037 [0,017; 0,070] 0,00 | 0,174 [0,090; 0,285] 0,95 | 0,218 [0,154; 0,283] 1,00 | 0,144 [0,078; 0,221] 0,89 |
| fs_a k=20, theo video | 0,075 [0,064; 0,092] 0,00 | 0,080 [0,069; 0,095] 0,00 | 0,078 [0,069; 0,092] 0,00 | 0,058 [0,026; 0,106] 0,05 | 0,082 [0,071; 0,104] 0,05 | 0,071 [0,057; 0,100] 0,02 |
| fs_b k=20, theo video | 0,085 [0,071; 0,101] 0,03 | 0,083 [0,072; 0,098] 0,01 | 0,078 [0,069; 0,092] 0,00 | 0,084 [0,050; 0,139] 0,31 | 0,083 [0,072; 0,104] 0,06 | 0,079 [0,063; 0,110] 0,15 |
| fs_b k=20, theo khung | 0,260 [0,059; 0,393] 0,63 | 0,190 [0,082; 0,274] 0,95 | 0,078 | 0,077 [0,048; 0,131] 0,24 | 0,086 [0,074; 0,108] 0,13 | 0,089 [0,044; 0,170] 0,49 |

Theo từng seed (`seed` 0/1/2):
- Worst_case C1: 0,205 / 0,062 / 0,234, với P > α là 0,63 / 0,03 / 0,63.
- Worst_case C5: P > α là 0,64 / 1,00 / 0,81.
- Pooled C3: P > α là 0,04 / 0,16 / 0,99, tức chỉ seed 2 vượt.

**Đọc.**
- Pooled theo khung: cận dưới > α ở C1, C4, C5, C6. C2 có 93 % mẫu vượt α nhưng khoảng vẫn chứa α (cận dưới 0,061). C3 không kết luận được.
- Khoảng ở C1 rất rộng (0,13–0,71) vì C1_seq11 (136/387 khung dương) chi phối.
- Pooled cấp cụm: chỉ C5 có khoảng nằm hẳn trên α. C4 (0,95) và C6 (0,89) có khả năng vượt; C1 (0,66) không kết luận được.
- Worst-case "vẫn vỡ ở C1 và C5": ở C1 không còn đứng được (P = 0,43, phụ thuộc seed); ở C5 khả năng cao (0,81) nhưng khoảng vẫn chứa α.
- Hiệu chuẩn lại 1 khung × 20 video: *trung bình* rủi ro theo video ≤ α được củng cố ở C1, C2, C3, C5 (P ≤ 0,06). Ở C4 (P = 0,31, cận trên 0,139; chỉ 34 cụm dương, test còn 14) và C6 (P = 0,15) thì không chắc.

## R2 — So sánh công bằng với in-domain (M5.2)
Từ `conformal_long.parquet` (`in_domain`: chia cụm 50/50, CRC theo ảnh, rủi ro theo khung) và
`conformal_cluster_long.parquet` (`in_domain_cl`, rủi ro theo video). Bảng: trung bình / q95 / max của 600 lần chia, và rủi ro
pooled (trung bình; min–max qua seed). Cột cuối: số seed (trên 3) có pooled > q95 in-domain *cùng seed*; trong ngoặc là số seed có cận
dưới bootstrap R1 > q95.

| Đích | in-domain theo khung | pooled | seed vượt q95 | in-domain_cl theo video | pooled_cl | seed vượt q95 |
|---|---|---|---|---|---|---|
| C1 | 0,116 / 0,236 / 0,290 | 0,468 (0,447–0,478) | 3 (0) | 0,095 / 0,161 / 0,246 | 0,113 | 0 (0) |
| C2 | 0,096 / 0,161 / 0,218 | 0,198 (0,194–0,203) | 3 (0) | 0,092 / 0,149 / 0,227 | 0,074 | 0 (0) |
| C3 | 0,135 / 0,324 / 0,453 | 0,100 (0,080–0,132) | 0 (0) | 0,090 / 0,180 / 0,265 | 0,037 | 0 (0) |
| C4 | 0,110 / 0,220 / 0,277 | 0,288 (0,226–0,343) | 3 (0) | 0,095 / 0,155 / 0,216 | 0,174 | 2 (0) |
| C5 | 0,113 / 0,224 / 0,291 | 0,375 (0,360–0,393) | 3 (3) | 0,091 / 0,158 / 0,222 | 0,218 | 3 (2) |
| C6 | 0,108 / 0,223 / 0,299 | 0,275 (0,231–0,345) | 3 (0) | 0,094 / 0,156 / 0,260 | 0,144 | 0 (0) |

**Đọc.**
- Lập luận "gấp 30–100 lần" **phải bỏ**: phản biện đúng.
- Phát biểu thay thế, theo từng fold: pooled theo khung vượt q95 in-domain cùng seed ở **C1, C2, C4, C5, C6 trong cả 3 seed**. Đây là so sánh một lần triển khai với đuôi trên của một lần chia, nên vẫn đứng được.
- Khi tính thêm sai số mẫu đích (R1), chỉ **C5** còn vượt chắc chắn.
- Ở cấp cụm, pooled_cl chỉ vượt q95 in-domain ở C5 (3/3) và C4 (2/3). Ở C1 và C6, rủi ro pooled_cl nằm trong dải mà một lần chia in-domain đơn lẻ cũng có thể cho ra.
- Lưu ý: lần chia in-domain chỉ hiệu chuẩn trên *một nửa* tập cal, nên q95 của nó rộng hơn so với một lần triển khai dùng toàn bộ cal. Vì vậy phép so sánh này nghiêng về phía bảo thủ (khó kết luận "vượt" hơn).

## R3 — Thành phần tập hiệu chuẩn nguồn (M1, M2; câu hỏi 2)
**Cách làm.** Pooled CRC theo ảnh, giữ tối đa m khung dương mỗi cụm cal:
- m ∈ {1, 5, 20}: lấy mẫu con ngẫu nhiên, R = 200 lần, dùng chung cho mọi seed.
- m = all: đúng bằng pooled đã công bố.
- Thêm biến thể bỏ từng trung tâm nguồn, và bỏ C2 + C4.

Ô: rủi ro theo khung ở đích (trung bình qua 200 mẫu con × 3 seed); trong ngoặc là tỷ lệ (mẫu con × seed) > α.

| Biến thể | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| tất cả khung (đã công bố) | 0,468 (1,00) | 0,198 (1,00) | 0,100 (0,33) | 0,288 (1,00) | 0,375 (1,00) | 0,275 (1,00) |
| ≤ 20 khung/video | 0,371 (1,00) | 0,192 (1,00) | 0,048 (0,00) | 0,167 (1,00) | 0,266 (1,00) | 0,188 (1,00) |
| ≤ 5 khung/video | 0,351 (1,00) | 0,190 (1,00) | 0,040 (0,00) | 0,155 (1,00) | 0,245 (1,00) | 0,170 (1,00) |
| 1 khung/video | 0,360 (1,00) | 0,185 (1,00) | 0,036 (0,00) | 0,149 (1,00) | 0,225 (1,00) | 0,159 (1,00) |
| bỏ C2 | 0,368 | — | 0,048 | 0,154 | 0,255 | 0,195 |
| bỏ C2 + C4 | 0,308 | — | 0,040 | — | 0,217 | 0,157 |
| bỏ C3 (toàn ảnh đơn) | 0,470 | 0,180 | — | 0,268 | 0,375 | 0,275 |
| bỏ C4 | 0,466 | 0,177 | 0,099 | — | 0,370 | 0,262 |

(bỏ C1 / C5 / C6 và IoU 0,5, α 0,20 có trong CSV; ở IoU 0,5 kết luận giống hệt.)

**Đọc — trả lời câu hỏi 2: có, thất bại pooled vẫn còn, nhưng nhỏ hơn đáng kể.**
- Với 1 khung/video, pooled vẫn vượt α ở C1, C2, C4, C5, C6 trong 100 % mẫu con × seed (khoảng 0,15–0,36, so với 0,20–0,47 đã công bố).
- Phần vượt α giảm 55–74 % ở C4, C5, C6:
  - C4: 0,188 → 0,049
  - C5: 0,275 → 0,125
  - C6: 0,175 → 0,059
- Phần vượt α giảm ít ở C1 (0,368 → 0,260) và C2 (0,098 → 0,085).
- Ở C3, kết quả "sát α" đã công bố (0,100) là do thành phần tập cal: với 1 khung/video thì còn 0,036.
- Nguyên nhân gần như hoàn toàn là **video C2 duy nhất (252 khung, dễ)**: chỉ riêng việc bỏ C2 đã cho kết quả gần bằng 1 khung/video. Video này làm ngưỡng pooled lỏng ra.
- Bỏ C3 (trung tâm toàn ảnh đơn, M2) gần như không đổi gì. Rò rỉ train/cal qua ảnh đơn không nối được ở C3 không phải là nguồn gốc của thất bại.
- Rủi ro theo video của biến thể 1 khung/video (C1 0,118; C4 0,175; C5 0,218; C6 0,145) gần bằng pooled_cl, nhất quán với thiết kế (b).
- Một lưu ý không hoàn toàn cùng điều kiện: sau khi cắt bớt, pooled ở C4 (0,149) và C6 (0,159) đã *thấp hơn* q95 in-domain của R2 (0,220 / 0,223). Tuy nhiên in-domain trong R2 vẫn dùng tập cal chưa cắt.

**Hệ quả cho bài.**
- Kết luận định tính "pooled nguồn hỏng ở 5/6 trung tâm" đứng vững.
- Các con số 0,20–0,47 phóng đại độ lớn. Kết quả chính nên dùng 1 khung/video hoặc pooled cấp cụm (như phản biện gợi ý), còn bản "mọi khung" nên báo như kết quả phụ, kèm giải thích vì sao nó tệ hơn.
- Yêu cầu M1(iii) (chia lại để mỗi trung tâm có đủ cụm dương trong vai trò cal) **không làm được** nếu không huấn luyện lại. Phải ghi rõ đây là hạn chế.

Hình (`fig_calibration_composition`). Draft caption: *Pooled-source CRC at the held-out centre when no single video dominates the source calibration set (IoU ≥ 0.1, α = 0.10). Realised frame-averaged miss rate using all calibration frames (as published), at most 20 or 1 positive frame per video (200 random subsamples), or with C2 and C4 removed from calibration; dashed line, α. Points are means over three training seeds (and subsamples); error bars show the range of the three per-seed means, not s.d. or s.e.m.*

## R4 — Lần rút degenerate và cảnh báo giả (M6, M9; câu hỏi 5)
**Nguồn.** Mọi lần rút có sẵn trong `conformal_cluster_long.parquet`:
- fs_a / fs_b với k = 10/20;
- fs_frames 20/50/100, đúng bằng `few_shot` của `conformal.py` (đã assert).

**Cách tính.**
- Kết quả của mỗi lần rút chia ba loại: *đạt* (không degenerate và rủi ro ≤ α), *vi phạm* (không degenerate và > α), *không chứng nhận được* (degenerate).
- FA = số box trên mỗi khung âm của tập test, tại λ̂.
- "any-alarm" = tỷ lệ khung âm có ≥ 1 box tại λ̂, tính trên *mọi* khung âm của đích. Đây là xấp xỉ: có gồm một ít khung âm nằm trong các cụm đã rút.

Bảng: 1 khung × 20 video (fs_b) và cụm k = 20 (fs_a), rủi ro theo video:

| Đích | thiết kế | degenerate | rủi ro mọi lần rút / chỉ không degenerate | đạt / vi phạm / không chứng nhận | FA trung bình mọi lần rút / không degenerate | FA trung vị [IQR] không degenerate | any-alarm trung vị |
|---|---|---|---|---|---|---|---|
| C1 | fs_b | 0,10 | 0,085 / 0,092 | 0,58 / 0,32 / 0,10 | 3,95 / 2,80 | 1,32 [0,37–4,27] | 0,66 |
| C2 | fs_b | 0,05 | 0,083 / 0,086 | 0,65 / 0,31 / 0,05 | — | — | — |
| C3 | fs_b | 0,04 | 0,078 / 0,081 | 0,67 / 0,29 / 0,04 | — | — | — |
| C4 | fs_b | 0,16 | 0,084 / 0,095 | 0,52 / 0,32 / 0,16 | 5,29 / 4,27 | 2,94 [1,75–4,56] | 0,74 |
| C5 | fs_b | 0,37 | 0,083 / **0,102** | 0,36 / 0,27 / 0,37 | 4,92 / 3,64 | 2,66 [1,58–5,23] | 0,76 |
| C6 | fs_b | 0,25 | 0,079 / 0,096 | 0,46 / 0,30 / 0,25 | 2,43 / 1,88 | 0,91 [0,29–2,78] | 0,47 |
| C4 | fs_a | 0,02 | 0,058 / 0,059 | 0,83 / 0,15 / 0,02 | 6,23 / 6,06 | 4,51 [3,21–8,06] | 0,89 |
| C5 | fs_a | 0,38 | 0,082 / 0,102 | 0,35 / 0,27 / 0,38 | 4,99 / 3,60 | 2,66 [1,58–5,23] | 0,76 |

Ngân sách theo khung (fs_frames, rủi ro theo khung), rủi ro trên lần rút không degenerate:
- C1 0,318 / 0,364 / 0,372 (20/50/100);
- C2 0,21–0,22;
- C6 0,15–0,18;
- C4 0,11–0,13.

Tất cả đều cao hơn trung bình gộp mọi lần rút.

**Đọc.**
- Phản biện đúng ở cả hai ý:
  - Lần rút degenerate (giữ mọi box) kéo trung bình rủi ro xuống và đẩy FA lên.
  - Chỉ tính trên lần rút không degenerate, rủi ro theo video của thiết kế (b) là 0,081–0,102, và **C5 vượt α (0,102)**.
- Tức là phát biểu "≤ 0,085 ở mọi trung tâm" một phần nhờ các lần rút "bỏ cuộc". Lưu ý thêm: điều kiện hoá trên "không degenerate" cũng nằm ngoài định lý CRC, vì đây là chọn lọc theo dữ liệu hiệu chuẩn.
- Báo kết quả theo ba loại: ở 20 video, tỷ lệ "không chứng nhận được" là 4–37 %, và trong số các lần còn chứng nhận được thì 27–32 % vẫn vi phạm.
- Về cảnh báo giả: FA trung vị trên lần rút không degenerate thấp hơn trung bình (phân phối lệch phải), 0,9–2,9 box/khung âm. Tuy nhiên **47–76 % khung không có polyp có ít nhất một cảnh báo**, và với fs_a ở C4 là 89 %.
- Với detector này, các điểm vận hành đạt bảo đảm đều gần như chắc chắn **không dùng được trong lâm sàng**. Bài phải nói thẳng điều này.
- Bài toán ràng buộc kép miss + FA (Learn-then-Test) chưa được thử. Đó là phương pháp mới, nằm ngoài scope freeze, nên không làm ở đây.

## R5 — AP chỉ trên khung dương và sàn bỏ sót theo ảnh (M7)
AP50 (IoU 0,5, như `eval_detection.py`), trung bình (min–max) qua 3 seed. "neg" = tỷ lệ khung âm trong tập.

| | val: neg / AP mọi khung / AP khung dương | đích: neg / AP mọi khung / AP khung dương |
|---|---|---|
| C1 | 0,54 / 0,704 / 0,747 | 0,58 / 0,509 (0,430–0,549) / 0,559 (0,501–0,597) |
| C2 | 0,66 / 0,713 / 0,752 | 0,13 / 0,695 / 0,706 |
| C3 | 0,59 / 0,708 / 0,738 | 0,00 / 0,814 / 0,814 |
| C4 | 0,46 / 0,797 / 0,816 | 0,79 / 0,461 (0,452–0,466) / **0,692** (0,668–0,706) |
| C5 | 0,57 / 0,755 / 0,786 | 0,78 / 0,489 (0,376–0,547) / 0,558 (0,451–0,621) |
| C6 | 0,45 / 0,716 / 0,750 | 0,82 / 0,679 / 0,715 |

Seed 0 ở C4 cho AP khung dương = 0,701, khớp Table S3 mà phản biện trích.

Sàn bỏ sót ở đích (giữ mọi box, λ = 0,001), IoU 0,1, trung bình qua seed (min–max):

| Đích | theo ảnh (cái CRC kiểm soát) | theo polyp (1 − max recall, Table 1) | theo video |
|---|---|---|---|
| C1 | 0,105 (0,041–0,142) | 0,110 (0,048–0,145) | 0,021 |
| C2 | 0,080 (0,071–0,087) | 0,093 | 0,021 |
| C3 | 0,015 (0,012–0,017) | 0,047 | 0,015 |
| C4 | 0,039 (0,025–0,053) | 0,051 | 0,030 |
| C5 | 0,055 (0,049–0,064) | 0,073 | 0,054 |
| C6 | 0,059 (0,038–0,071) | 0,075 | 0,040 |

Ở IoU 0,5, sàn theo ảnh / theo polyp: C1 0,242/0,250; C2 0,129/0,145; C3 0,027/0,063; C4 0,102/0,117; C5 0,151/0,176; C6 0,108/0,126.

**Đọc.**
- **Chênh lệch AP phần lớn do tỷ lệ khung âm ở C4**: AP khung dương 0,692, so với 0,461 khi tính mọi khung.
- Tính trên khung dương, "AP giảm khi đổi bệnh viện" vẫn đúng:
  - giảm rõ ở C1 (0,747 → 0,559), C5 (0,786 → 0,558), C4 (0,816 → 0,692);
  - giảm ít ở C2 và C6;
  - tăng ở C3.
- **Sàn theo ảnh** (đúng đại lượng CRC kiểm soát) là 0,015–0,105, thấp hơn sàn theo polyp vì ảnh nhiều polyp bị bỏ sót nhiều hơn. Điều này giải quyết mâu thuẫn ở C2: worst-case degenerate cho 0,080, đúng bằng sàn theo ảnh 0,080.
- α = 0,05 *theo khung* chỉ không đạt được (do sàn) ở C1, C2, C6, và C5 ở mức sát biên. Ở C3 và C4 thì về nguyên tắc đạt được.
- Sàn theo video thấp hơn nữa (0,015–0,054).
- Câu "sàn 5–11 % làm α = 0,05 không khả thi" phải sửa thành dùng sàn theo ảnh và chỉ rõ từng trung tâm.

## Phán quyết theo từng nhận xét
- **M1 (thành phần cal):** đúng một phần. Hướng kết luận đứng vững: pooled vẫn hỏng ở 5/6 trung tâm khi chỉ dùng 1 khung/video và khi bỏ C2 + C4. Độ lớn **giảm mạnh**: 0,15–0,36 thay vì 0,20–0,47, và 55–74 % phần vượt α ở C4–C6 đến từ một video C2 duy nhất. Kết quả C3 "sát α" là do thành phần cal. Yêu cầu (iii) không làm được.
- **M2 (rò rỉ qua ảnh đơn):** bỏ C3 không đổi kết luận, nên kết luận đứng vững. Recall của quy tắc nối ảnh và câu hỏi về Corbetta chưa trả lời (không có trong R1–R5).
- **M4 (số seed):** chưa giải quyết được: không có seed mới do không dùng GPU và scope đã chốt. Khoảng phân cấp R1 đã đưa biến thiên của 3 seed vào. Các phát biểu theo đa số seed thì **đổ**: worst-case C1 (P > α theo seed là 0,63/0,03/0,63) và pooled C3 (chỉ seed 2 vượt).
- **M5.1 (sai số mẫu đích):** đã bổ sung R1. Với pooled theo khung, kết luận "vượt" có cận dưới > α ở C1, C4, C5, C6; C2 chỉ là khả năng cao (P 0,93). Pooled cấp cụm chỉ vượt chắc ở C5. Câu "hỏng ở 5/6" **yếu đi**: phải viết lại thành 4 trung tâm chắc chắn, 1 khả năng cao, 1 không kết luận được.
- **M5.2 ("30–100×"):** lập luận **đổ**. Thay bằng so sánh với q95 in-domain theo từng fold: vượt ở 5/6 trung tâm trong cả 3 seed (theo khung). Ở cấp cụm, chỉ C5 vượt (C4 vượt ở 2/3 seed).
- **M6 (degenerate):** phản biện đúng. Rủi ro trên lần rút không degenerate là 0,081–0,102, C5 > α. Phải báo kết quả theo ba loại. Câu "≤ 0,085 ở mọi trung tâm" **yếu đi**.
- **M7:**
  - AP: kết luận đứng vững nếu dùng AP khung dương, nhưng mức giảm ở C4 nhỏ hơn nhiều.
  - Sàn bỏ sót: định nghĩa phải **sửa** (dùng sàn theo ảnh 1,5–10,5 %). Khẳng định α = 0,05 không khả thi chỉ đúng ở C1, C2, C6 (và C5 sát biên).
  - Khuyến nghị "hiệu chuẩn lại tại chỗ" **đổ** nếu không kèm cảnh báo: 47–89 % khung không polyp có cảnh báo. Phải nói rõ là bảo đảm chỉ đạt được ở điểm vận hành khó dùng trong lâm sàng.
- **M9:** đã thêm tỷ lệ khung âm có cảnh báo (any-alarm, xem R4), và nó củng cố nhận định "không dùng được". Tiêu chí tạm thời theo tổn thương và mô hình chi phí gán nhãn chưa làm.

## Hạn chế của chính các phân tích này
- R1, hiệu chuẩn lại: mỗi mẫu bootstrap chỉ có 100 lần rút (nhiễu Monte Carlo của trung bình khoảng ≤ 0,005). Mẫu bootstrap ít video phân biệt hơn mẫu gốc, nên ở C4 tập test nhỏ hơn thực tế và khoảng có xu hướng rộng (bảo thủ). Chỉ có 3 seed nên phần "huấn luyện" của khoảng phân cấp rất thô.
- R2: so sánh một lần triển khai dùng *toàn bộ* cal với lần chia in-domain dùng *nửa* cal.
- R4: any-alarm tính trên mọi khung âm của đích (xấp xỉ). FA theo box lấy đúng từ parquet.
- Không phân tích nào thay đổi detector, các lần chia hay dữ liệu. Mọi số trong ghi chú này lấy từ các CSV ở trên.
