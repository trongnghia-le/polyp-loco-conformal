# 2026-11-08 — Vẽ lại Fig. 2–5 theo bản sửa (revision round 1)

Mục đích: đồ hoạ khớp với văn bản đã sửa và các góp ý về hình của phản biện (`paper/review/review_round1.md`, Minor 5, 14–17; M5.5). Không train, không thí nghiệm mới; chỉ đọc `outputs/eval/*.csv`. Số hình chính giữ nguyên (5 hình + 3 bảng = 8).

Kiểm tra chung cho cả 4 hình: `configs/paper.mplstyle` (Arial 7/6.5 pt, TrueType — `FontFile2` có trong mỗi PDF, không có Type 3), rộng 183 mm, mọi nét ≥ 1 pt, nền trắng; xuất `.pdf/.eps/.png` vào `results/figures/`; đã render PNG và xem ở kích thước cuối, lặp đến khi không còn chồng chữ/che dữ liệu.

## Fig. 2 — `scripts/fig_risk_vs_alpha.py` → `fig_risk_vs_alpha.*`
- Trục y "Realized miss rate (target)" → **"Realised frame-averaged miss rate"** (Minor 5, Minor 14: chính tả UK, gọi đúng đại lượng như văn bản).
- Key "Target few-shot (50)" → **"Frame budget, 50 frames"** (Minor 5: thống nhất tên với Table 2 "Frame budget, 20/100 frames").
- Cỡ chữ (Minor 14): hình vẽ đúng 183 mm nên chữ ở bản in là 7 pt (nhãn tick/key 6.5 pt) — không đổi, giữ thống nhất với Fig. 1/S1/S2.
- **Không đổi**: error bar vẫn là range của 3 seed (M5.5, Minor 14 gạch 1). Lý do: mean ± s.d. hay percentile interval có nghĩa cần thêm seed (ngoài scope freeze); khoảng bootstrap R1 (`revision_R1_target_bootstrap.csv`) chỉ có cho pooled / worst case, không có cho in-domain và frame budget, nên trộn vào một hình sẽ cho hai loại error bar khác nhau. Legend đã trỏ tới Table 2 cho khoảng tin cậy. Nếu tác giả muốn đổi sang khoảng R1 cho hai chuỗi pooled/worst case thì dữ liệu đã có.
- Câu giải thích in-domain > α ở C3 đã có sẵn trong legend.
- Caption nháp: *Realised frame-averaged miss rate at the held-out centre versus α for four calibration strategies; points above the diagonal miss the target.*

## Fig. 3 — `scripts/fig_fewshot_budget.py` → `fig_fewshot_budget.*`
- Key "20/50/100 target images" → **"20/50/100 frames"** (Minor 5, Minor 15).
- Đường in-domain: trước là nét đứt xám, khó thấy trên cột xanh (Minor 15). Nay là **đoạn đen liền có viền trắng (halo) + hình thoi trắng viền đen** ở mép phải nhóm cột; error bar vẽ đè lên đoạn này để không bị cắt.
- Trục y "Violation frequency (risk > α)" → **"Violation frequency (frame-averaged miss rate > α)"**.
- Legend: bỏ câu giải thích "target images", "Dashed segments" → "Black segments with open diamonds".
- Caption nháp: *Share of frame-budget recalibrations whose realised frame-averaged miss rate exceeded α, by budget and held-out centre, with the in-domain reference.*

## Fig. 4 — `scripts/fig_cluster_budget.py` → `fig_cluster_budget.*`
(Script có thay đổi local chưa commit từ trước; bản mới viết lại trên nền bản đó, giữ nguyên dữ liệu, kiểu đánh dấu và màu của hai thiết kế theo video.)
- Minor 16 gạch 2: chuỗi frame budget (xanh lá, đặt ở "median k") **tách sang panel riêng g (α = 0.10) và h (α = 0.20)**: trục x = trung tâm held-out, ba mức 20/50/100 positive frames theo thang xanh aqua thứ bậc (`#5cc49b, #1baf7a, #0e6b4a`, đã chạy `validate_palette.js --ordinal`: pass), cùng thang y "mean video-averaged miss rate − α". Không còn trộn thiết kế tính theo frame với trục số video.
- Minor 16 gạch 1 (dày đặc): a–f nay chỉ còn 2 chuỗi; nhãn trục x chung một lần ("Labelled positive target videos, k (logarithmic axis)"), nhãn y chung; trục x thu về 3.5–55 (không cần chỗ cho điểm frame budget ở k ≈ 100) với tick 5/10/20/40.
- Trục y nêu rõ **"Mean video-averaged miss rate − α"** (trước "Mean risk − α").
- Key theo Table 2: "Video budget (cluster-level CRC, all frames of k videos)", "One frame × k videos (image-level CRC)".
- Minor 16 gạch 3 (degenerate draws nằm trong mean) đã có trong legend từ trước; giữ.
- Số liệu thấy ở g (α = 0.10, `conformal_cluster_summary.csv`, fs_frames, seed all, risk_cl_mean): C4 0.127/0.135/0.122, C6 0.126/0.123/0.108 với 20/50/100 frames → frame budget vượt α trên thang video-averaged ở C4 và C6, các trung tâm khác dưới hoặc sát α.
- Kích thước 183 × 128 mm (trước 183 × 90 mm).
- Caption nháp: *Mean video-averaged miss rate minus α after recalibration at the held-out centre: budgets counted in videos (a–f) and in frames (g, h).*

## Fig. 5 — `scripts/fig_clinicdb.py` → `fig_clinicdb.*`
- Minor 17: hai dải 5–95 % chồng nhau và hai đường mean không phân biệt được. Panel a nay vẽ **trực tiếp hiệu và độ phân tán**: trên — mean realised frame-averaged miss rate − α (error bar = s.d. qua 18 detector); dưới — s.d. của realised frame-averaged miss rate qua mọi draw × detector (`risk_fr_sd_draws`). Ở α = 0.10, ~310 frame: mean 0.098 (frame split) vs 0.102 (video split), s.d. qua draw 0.023 vs 0.050 (`clinicdb_summary.csv`, run all, budget 300) — khớp Results.
- Giữ nhãn a/b vì Results trích Fig. 5a (within) và Fig. 5b (external).
- Trục y: "Realised … (frame-averaged)" và **"Realised video-averaged miss rate"** ở b.
- Legend: thay câu mô tả dải 5–95 % bằng mô tả hai trục của a; ghi rõ s.d. qua detector là biến thiên giữa các lần train, không phải khoảng tin cậy.
- Caption nháp: *ClinicDB: (a) within-dataset frame vs video split, mean excess over α and spread over calibration draws; (b) external validation of the PolypGen-calibrated and recalibrated thresholds.*

## Miss floor
Không hình nào trong Fig. 2–5 vẽ đường miss floor (cả trước và sau) nên không cần áp định nghĩa image-level floor (`revision_R5_detection_floor.csv`). Không thêm floor vào hình vì phản biện không yêu cầu.

## File đã sửa
- `scripts/fig_risk_vs_alpha.py`, `scripts/fig_fewshot_budget.py`, `scripts/fig_cluster_budget.py`, `scripts/fig_clinicdb.py`
- `results/figures/fig_{risk_vs_alpha,fewshot_budget,cluster_budget,clinicdb}.{pdf,eps,png}`
- `paper/sections/legends.tex` (chỉ legend Fig. 2–5 và comment đầu file; ≤ 240 từ mỗi legend)
- `results/figures/README.md` (4 dòng)
- Biên dịch: `scripts/make_overleaf_zip.py` → main.pdf 40 trang, supplementary.pdf 10 trang, undefined citations: none, undefined refs: 0.

## Việc còn mở (cho tác giả)
- `paper/review/response_round1.md` vẫn ghi "figures not regenerated" ở M5.5, Minor 5, 14–17 — cần sửa lại thành "done" (không sửa ở đây theo phạm vi nhiệm vụ).
- Quyết định có đổi error bar Fig. 2 sang khoảng bootstrap R1 cho pooled/worst case hay không.
