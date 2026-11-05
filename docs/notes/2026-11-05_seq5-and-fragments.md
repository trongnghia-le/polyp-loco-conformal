# 2026-11-05 — seq5 thuộc C2; mask "vỡ mảnh" là do dụng cụ cắt ngang polyp

## seq5 (bài báo ghi C5)
Bằng chứng (`scripts/audit_seq5.py`, `scripts/audit_seq5_deep.py`, ảnh trong `outputs/seq5_pairs.jpg`, `outputs/seq5_overlay_crops.jpg`):
- Ảnh đơn C2 `EndoCV2021_001351/1352/1353` gần trùng pixel với seq5 khung 12/92/200 (lệch TB 1,8–3,1 mức xám); xem cạnh nhau: cùng polyp, cùng vị trí dụng cụ, đồng hồ trên ảnh lệch ≤ 1 giây → cùng một video.
- Lớp chữ trên ảnh của seq5 (`■■■/■■■(0/1)`, `Eh:A3 Cm:1`, ngày `/2017`) giống hệt ảnh đơn và chuỗi C2; ảnh C5 (đơn và chuỗi âm) không có lớp chữ này.
- Bảng lượng tử JPEG giống hệt nhau ở mọi trung tâm (bộ dữ liệu đã được nén lại đồng loạt) → phép thử này không phân biệt được nguồn.
- Kết luận: seq5 cùng nguồn với C2; Fig. 7 và tổng số trong văn bản của bài gốc gán nhầm sang C5. Lỗi thứ ba của bài gốc.
- Đề xuất: seq5 → C2 (C2 chuỗi dương 1.115, C5 chuỗi dương 0). **Chờ người dùng chốt.**

### Kiểm lại lần 2 (`scripts/audit_seq5_nn.py`)
- So mọi ảnh đơn của 6 trung tâm với mọi khung của 23 chuỗi dương (ảnh xám 64×48, MAD < 6): ảnh đơn C2 mang số **001318–001353 được lấy tuần tự từ 5 video**: seq12 (001318–1320), seq13 (001334–1335), seq14 (001342–1344), seq15 (001140, 001346) và **seq5 (001351–1353, MAD 0,75–2,3)**. C6 cũng có ảnh đơn lấy từ chuỗi C6 (seq16, 17, 21, 22). **Ảnh đơn C5 không trùng với khung nào của bất kỳ chuỗi nào**, kể cả seq5.
- Zip "Concatenated" lồng bên trong: seq5 vẫn là thư mục `seq5/` riêng, không có thông tin trung tâm.
- Table 3 của bài: seq5 "5 mm polyp catched by a snare; partial occlusion by biopsy instrument" — cùng kiểu thủ thuật với seq13 ("6–7 mm polyp catched with a snare") và seq14 (snare quanh cuống) của C2.
- Giả thuyết ngược (seq5 đúng là C5, còn 3 ảnh đơn bị xếp nhầm vào C2) không khớp: 3 ảnh đó nằm giữa dải số tuần tự 0013xx toàn lấy từ video C2, cùng độ phân giải 1440×1064 và lớp chữ Olympus của C2.
- Kết luận chắc chắn: seq5 là một video của nguồn C2.

### Mask vỡ mảnh — kiểm lại
- 12 khung chọn ngẫu nhiên trong 96 ca "mọi mảnh nằm trong bbox": **12/12 là sợi thòng lọng (dây xanh) hoặc kẹp kim loại cắt ngang polyp**, mask chừa dụng cụ ra (quy ước vẽ mask đúng, không phải lỗi). Ảnh: `outputs/fragments_random12.jpg`.
- Có ca bbox chính thức cũng chia polyp làm 2 khung (seq14_732) → hiếm; theo bbox chính thức.

## Mask nhiều mảnh (`scripts/audit_fragments.py`, `outputs/fragments*.csv|jpg`)
- 305 khung có ≥ 2 vùng mask (sau ngưỡng 50 px). Phần lớn là **nhiều polyp thật** (ca đa polyp C3 tới 6–15 polyp; bbox chính thức cũng nhiều khung). → Không được gộp theo khoảng cách: gộp ở d = 20 px làm sai 31 khung có nhiều polyp thật.
- 99 khung có số vùng mask > số bbox chính thức (chủ yếu seq14 49, seq5 24, seq13 9 — video cắt polyp). 96/99 có mọi vùng nằm trong một bbox chính thức: **thòng lọng/dụng cụ cắt ngang polyp, mask chừa sợi dây ra nên polyp bị tách mảnh**; bbox chính thức bao trọn polyp.
- 3 khung có vùng nằm ngoài mọi bbox: `data_C2/images_C2/EndoCV2021_001321.jpg` (mảnh lệch nhẹ khỏi bbox, có vẻ vẫn là cùng polyp), `data_C3/images_C3/C3_EndoCV2021_00368.jpg` (2 vùng nhỏ không có bbox, có thể là polyp bị sót khung), `data_C5/images_C5/EndoCV2021_C5_50000109.jpg` (đốm thứ hai dưới polyp, không rõ). → nhờ chuyên gia xem.
- Đề xuất quy tắc: khi có bbox chính thức, **mỗi bbox chính thức = 1 polyp** (các vùng mask nằm trong cùng bbox thuộc cùng polyp; GT box cho detection = bbox chính thức); vùng mask nằm ngoài mọi bbox tính là polyp riêng (thận trọng) và đánh dấu; khi không có file bbox (64 ảnh C3) dùng vùng mask. Mask vẫn là GT cho phân vùng.

## Hình
Chưa có hình cho bài; ảnh kiểm tra nằm trong `outputs/` (không đưa vào bài). Ứng viên hình phụ lục: một ô "polyp bị thòng lọng tách mask" để giải thích quy tắc đếm.
