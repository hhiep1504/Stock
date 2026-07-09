
# Thống kê Biến động Lợi nhuận (Min/Max Return)

### 1. Tổng quan

File này tổng hợp mức độ biến động lợi nhuận cực đại và cực tiểu của 26 mã cổ phiếu bất động sản theo từng **Quý** trong giai đoạn 2019 - 2024.

- **Mục đích:** Giúp nhà đầu tư đánh giá rủi ro (đáy thấp nhất) và tiềm năng (đỉnh cao nhất) của từng mã trong một khoảng thời gian cụ thể (3 tháng).

### 2. Cấu trúc File (Multi-level Header)

Dữ liệu này có cấu trúc cột phức tạp hơn (Header đa cấp) so với file giá cổ phiếu thông thường.

|**Cấp độ**|**Chi tiết**|
|---|---|
|**Cột Index**|`Quy` (Quý 1, 2, 3, 4) và `Nam` (2019 - 2024)|
|**Cột Level 0 (Nhóm)**|Chia làm 2 nhóm lớn: **Min Return** và **Max Return**|
|**Cột Level 1 (Chi tiết)**|Tên các mã cổ phiếu (CCL, VHM, NVL...) nằm dưới mỗi nhóm|

**Minh họa trực quan:**

|**Quy**|**Nam**|**Min Return **||**...**|**Max Return **||
|---|---|---|---|---|---|---|
|||_CCL_|_CDC_|...|_CCL_|_CDC_|
|3|2019|-0.09|#DIV/0!|...|0.72|#DIV/0!|
|4|2019|-0.34|-0.21|...|-0.07|0.16|

### 3. Giải thích Ý nghĩa Chỉ số

- **Min Return (Lợi nhuận Tối thiểu):** Mức giảm giá sâu nhất (hoặc tăng ít nhất) của cổ phiếu trong quý đó so với giá mở cửa đầu quý.
    - _Ví dụ:_ `-0.34` nghĩa là có lúc cổ phiếu đã giảm **34%**. Đây là thước đo **Rủi ro (Risk)**.
- **Max Return (Lợi nhuận Tối đa):** Mức tăng giá cao nhất của cổ phiếu trong quý đó.
    - _Ví dụ:_ `0.72` nghĩa là có lúc cổ phiếu đã tăng **72%**. Đây là thước đo **Tiềm năng (Potential)**.
