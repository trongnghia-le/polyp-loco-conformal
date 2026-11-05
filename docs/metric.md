# Định nghĩa metric (đóng băng trước khi chạy thí nghiệm)

Mọi thay đổi sau khi đã nhìn kết quả phải ghi thành mục "Thay đổi" cuối file, kèm lý do. Cài đặt tham chiếu: `scripts/metrics.py` (numpy, có self-check). Ngưỡng nhị phân hoá mask và diện tích tối thiểu nằm ở `configs/data.yaml`, khoá `mask_threshold` và `min_area`.

## 1. Đơn vị polyp
Một **polyp** trong một khung = một connected component (8-connectivity) của `mask > mask_threshold`, bỏ component có diện tích < `min_area` pixel. Box GT = bounding box chặt của component, `[x0, y0, x1+1, y1+1]` (nửa mở, pixel ảnh gốc). Box GT luôn sinh từ mask, kể cả 64 ảnh C3 thiếu file bbox, để cả sáu trung tâm dùng cùng một quy tắc. Ảnh có 0 component sau lọc là **ảnh âm** (gồm 5/31/1/81/2/5 ảnh mask rỗng ở C1..C6 và mọi khung `seq*_neg`). Box dự đoán được đổi về pixel ảnh gốc trước khi tính IoU.

## 2. Phát hiện ở cấp khung
Với ngưỡng điểm λ, tập dự đoán là P(λ) = {box có score ≥ λ}. Một polyp g được **phát hiện** nếu được ghép với một box p ∈ P(λ) có IoU(p, g) ≥ 0.5.

**Quy tắc ghép: một-một, tham lam theo score giảm dần** (hoà thì theo thứ tự đầu vào). Duyệt p theo score giảm dần; p nhận GT chưa ghép có IoU cao nhất nếu IoU ≥ 0.5, ngược lại không ghép gì.
- Vì sao một-một: nếu một box lớn được phép "phủ" hai polyp sát nhau thì mỗi polyp đều tính là phát hiện, làm tỷ lệ bỏ sót thấp giả. Một-một đúng với ý nghĩa lâm sàng "mỗi polyp cần một cảnh báo".
- Đơn điệu theo λ: danh sách được duyệt theo score giảm dần nên các quyết định ghép ở λ' > λ chính là tiền tố của các quyết định ở λ (bỏ box điểm thấp không đổi quyết định của box điểm cao hơn). Do đó tập polyp được phát hiện ở λ' là tập con của tập ở λ, và loss ở mục 3 **không giảm khi λ tăng** theo từng ảnh. CRC cần đúng tính chất này (loss đơn điệu theo tham số). Cách "một box bất kỳ có IoU ≥ 0.5" không loại trừ cũng đơn điệu nhưng cho phép phủ đôi; ghép Hungarian tối đa hoá số cặp cũng có tổng đơn điệu nhưng không đơn điệu theo từng polyp và đắt hơn, nên không dùng.

## 3. Loss và rủi ro mà CRC kiểm soát
Với ảnh i có K_i ≥ 1 polyp:

L_i(λ) = (số polyp không được phát hiện ở λ) / K_i ∈ [0, 1], B = 1.

Ảnh có K_i = 0: loss quy ước 0 trong code nhưng **loại khỏi hiệu chuẩn và khỏi rủi ro**. Lý do: tỷ lệ ảnh âm khác nhau mạnh giữa các trung tâm (C2, C3 không có chuỗi âm), nên trộn vào sẽ làm α không so sánh được; cảnh báo giả được báo riêng ở mục 5.

Rủi ro: R(λ) = E[L(λ)] trên phân phối ảnh dương của trung tâm đích. Đây là trung bình của phân số theo ảnh, khác tỷ lệ bỏ sót theo thể hiện (Σ bỏ sót / Σ polyp); báo thêm số thứ hai như chỉ số phụ, không được CRC bảo đảm.

**Chọn ngưỡng** (Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, arXiv:2208.02814, ICLR 2024). Bài gốc giả sử loss không tăng theo λ; ở đây loss không giảm theo λ nên áp dụng với λ := −λ. Với n ảnh hiệu chuẩn, R̂_n(λ) = (1/n) Σ L_i(λ):

λ̂ = sup { λ ∈ Λ : (n/(n+1)) · R̂_n(λ) + B/(n+1) ≤ α } = sup { λ : (Σ_i L_i(λ) + 1)/(n+1) ≤ α }

Nếu tập rỗng thì λ̂ = min Λ (cho mọi box qua) và lần rút bị đánh dấu **degenerate**. Với ảnh test mới trao đổi được với ảnh hiệu chuẩn: E[L_{n+1}(λ̂)] ≤ α. Đây là bảo đảm **kỳ vọng**, không phải cho từng lần rút; vì vậy mục 7 báo cả tần suất lần rút có rủi ro thực > α. Lưới Λ: 0.001..0.999 bước 0.001; dự đoán lưu với conf floor ≤ 0.001 (khoá cấu hình đặt khi chạy suy luận). Bảo đảm chỉ có nghĩa nếu detector ở λ = min Λ có R ≤ α; nếu không, λ̂ = min Λ và lần rút được đếm là degenerate (không phải lỗi của CRC).

Cỡ mẫu tối thiểu: ngay cả khi R̂ = 0 vẫn cần 1/(n+1) ≤ α, tức **n ≥ 1/α − 1**: α = 0.05 → 19, 0.10 → 9, 0.20 → 4. Ở 20 ảnh và α = 0.05, điều kiện thành Σ L_i ≤ 0.05, tức gần như không được bỏ sót ảnh nào.

## 4. Cấp chuỗi (C1, C2, C4, C5, C6; C3 không có chuỗi)
Dữ liệu không có track ID. **Giả định A1: mỗi thư mục `seqK` dương tính chứa đúng một polyp xuyên suốt, nên track = (thư mục chuỗi)**. Kiểm chứng trước khi dùng, ghi kết quả vào `docs/data_audit.md`: (i) tỷ lệ khung có đúng 1 component sau lọc (cần gần 100%); (ii) sắp theo số khung trong tên file, IoU/khoảng cách tâm giữa hai khung liên tiếp không có bước nhảy lớn; (iii) xem contact sheet các khung có ≥ 2 component. Nếu A1 sai ở một chuỗi, tách track bằng nối tham lam theo IoU giữa các khung liền kề và báo số chuỗi bị ảnh hưởng. Khung được đánh số lấy mẫu thưa (ví dụ 169, 320), nên "khung" ở đây là **khung có chú thích**.

Một track được **phát hiện cấp chuỗi** nếu ở ít nhất một trong **N = 5 khung có chú thích đầu tiên** (theo số khung tăng dần) polyp được phát hiện theo mục 2 ở λ̂ đã chọn. Độ nhạy: N ∈ {1, 3, 10}. N = 5 là giá trị đề xuất, cần người dùng xác nhận trước khi chạy. Chỉ số này **phụ thuộc λ̂ nhưng không được CRC bảo đảm**; chỉ báo mô tả (C1 và C5 chỉ có một chuỗi dương tính, C2 bốn, C4 chín thư mục nhưng 5 tiền tố nguồn), không báo tần suất vi phạm.

## 5. Cảnh báo giả
Trên khung âm (mục 1): FA(λ) = số box có score ≥ λ trong khung. Báo trung bình mỗi khung và trên 100 khung âm, tại λ̂. Khung âm gồm chuỗi âm và ảnh đơn mask rỗng; **chỉ C1, C4, C5, C6 có chuỗi âm** (C2, C3 chỉ có vài ảnh đơn rỗng, nên FA ở đó chỉ mô tả, không so sánh). Khung âm liên tiếp tương quan mạnh, khoảng tin cậy phải bootstrap theo cụm (mục 6).

## 6. Đơn vị trao đổi được: cụm, không phải khung
Các khung cùng video gần như trùng nhau, nên rút khung ngẫu nhiên làm ảnh hiệu chuẩn và ảnh test lẫn từ cùng video; khi đó hai tập không độc lập và bảo đảm đo được sẽ lạc quan giả. Vì vậy mọi lần rút hiệu chuẩn/test chia theo `cluster_id` trong `data/clusters.csv` (ảnh đơn hoặc nhóm theo tiền tố; video nguồn theo tiền tố, không theo thư mục, vì seq1/seq2, seq3/seq4, seq6/seq10, seq7/seq8 chung nguồn). Một lần rút: xáo trộn cụm của trung tâm đích, thêm nguyên cụm vào tập hiệu chuẩn cho tới khi số **ảnh dương** ≥ ngân sách; mọi cụm còn lại là tập test (rời cụm). Số ảnh thực n và số cụm k của mỗi lần rút được ghi lại; k mới là cỡ mẫu độc lập hiệu dụng. Trung thực về giới hạn: CRC ở đây chạy trên ảnh, còn đơn vị trao đổi là cụm, nên bảo đảm lý thuyết chỉ gần đúng; chính tần suất vi phạm đo ở mục 7 là phép kiểm tra thực nghiệm của điều đó. Ngân sách mà một cụm đơn đã vượt (ví dụ cụm video lớn ở C6) vẫn được giữ, với k = 1 được báo rõ.

## 7. Báo cáo theo từng trung tâm held-out
Không gộp trung tâm. Với mỗi trung tâm, α ∈ {0.05, 0.10, 0.20}, ngân sách ∈ {20, 50, 100} ảnh dương, R = 1000 lần rút theo cụm (seed cố định):
1. **Tỷ lệ bỏ sót thực** trên tập test: trung bình L_i (ảnh dương), so với α; trung bình qua các lần rút (cần ≤ α) và khoảng phân vị 5-95%. Phụ: tỷ lệ bỏ sót theo thể hiện.
2. **Tần suất vi phạm** = tỷ lệ lần rút có rủi ro test > α.
3. **Cảnh báo giả** tại λ̂: mỗi khung âm và mỗi 100 khung âm (chỉ C1, C4, C5, C6).
4. **Tỷ lệ lần rút degenerate**: λ̂ = min Λ (không ngưỡng nào đạt, kể cả do n < 1/α − 1 hoặc detector không đủ). Báo thêm tỷ lệ λ̂ = max Λ (α quá lỏng). Ở 20 ảnh với α = 0.05, n chỉ nhỉnh hơn mức tối thiểu 19, nên tỷ lệ này dự kiến cao.
5. Cấp chuỗi (mục 4), mô tả, kèm số chuỗi.
Các con số kèm n trung bình, k trung bình của lần rút.

## Quyết định cần người dùng xác nhận
- Ghép một-một tham lam (mục 2) thay vì "bất kỳ box nào".
- Ảnh âm loại khỏi rủi ro CRC (mục 3).
- Giả định A1 (một polyp/chuỗi) và N = 5 khung có chú thích (mục 4).
- Rút theo nguyên cụm tới khi đủ ngân sách ảnh (n thực có thể vượt ngân sách) (mục 6).
