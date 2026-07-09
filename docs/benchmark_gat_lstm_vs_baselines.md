**Dự án:** So sánh mô hình GAT-LSTM với các baseline trên tập dữ liệu thị trường chứng khoán  
**Ngày chạy:** 29/03/2026  
**Môi trường:** Kaggle Notebook, GPU (CUDA)

---

## 1. Tổng Quan

Notebook thực hiện benchmark so sánh hiệu suất của mô hình **GAT-LSTM (Graph Attention Network + LSTM)** — mô hình đề xuất — với một loạt các baseline, bao gồm các mô hình machine learning truyền thống (Sklearn) và các mô hình deep learning dạng chuỗi/đồ thị. Mục tiêu là đánh giá xem GAT-LSTM có thực sự vượt trội hơn các phương pháp chuẩn hay không, đặc biệt trong bài toán dự đoán `min_return` và `max_return` của cổ phiếu theo tuần.

---

## 2. Dữ Liệu

### 2.1 Nguồn dữ liệu

|File|Mô tả|
|---|---|
|`stock_market_19_24.csv`|Dữ liệu giá cổ phiếu ngày, giai đoạn 2019–2024|
|`min_max_return.csv`|Nhãn mục tiêu: min/max return theo tuần|

### 2.2 Pipeline xử lý dữ liệu

Dữ liệu được xử lý qua các bước sau:

1. **Load dữ liệu ngày** (`DataLoader`) → danh sách mã cổ phiếu (`stock_codes`)
2. **Feature Engineering** (`FeatureEngineer`) với `aggregation_mode = "weekly"`:
    - Tính 4 nhóm đặc trưng: `feat_std`, `feat_mean`, `feat_return`, `feat_skew`
    - Tính nhãn: `target_min`, `target_max`
    - Tạo tensor `x_full [T, N, F]` và `y_full [T, N, 2]`
3. **Xây dựng đồ thị** (`GraphConstructor`):
    - Dùng sector mapping để tạo `static_edge_index`
    - Đồ thị tĩnh, không thay đổi theo thời gian
4. **Tạo sequences** (`create_sequences`):
    - `window_size = 8` tuần → mỗi sample là một chuỗi 8 tuần
    - Mỗi phần tử sequence: `(x_seq [8, N, F], y_target [N, 2], edge_index)`
    - `top_k = 5`, `similarity_metric = "cosine"`, `corr_threshold = 0.7`

### 2.3 Shape dữ liệu

```
x_full:     [T_weeks, N_stocks, F_features]
y_full:     [T_weeks, N_stocks, 2]        # 2 = (min_return, max_return)
sequences:  List of (x_seq, y_target, edge_index)
            x_seq:      [8, N, F]
            y_target:   [N, 2]
            edge_index: [2, E]
```

---

## 3. Chiến Lược Phân Chia Dữ Liệu

### 3.1 Expanding Window (Walk-Forward Validation)

Notebook sử dụng chiến lược **Expanding Window** — một dạng Walk-Forward phù hợp cho dữ liệu chuỗi thời gian tài chính. Đây **không phải K-Fold thông thường** (không trộn thứ tự thời gian).

**Nguyên tắc:**

- Tập train luôn **chỉ sử dụng dữ liệu quá khứ** so với tập test
- Tập train **mở rộng dần** sau mỗi window
- Tập test là một đoạn cố định ngay sau tập train

**Cấu hình cụ thể:**

|Tham số|Giá trị|Ý nghĩa|
|---|---|---|
|`min_train_size`|150 sequences|Train tối thiểu ~150 tuần (~3 năm) trước khi bắt đầu test|
|`test_step`|15 sequences|Mỗi window test 15 tuần (~1 quý), sau đó mở rộng train thêm 15 tuần|

**Ví dụ với N = 200 sequences tổng:**

```
Window 1: Train [0–150]   → Test [150–165]
Window 2: Train [0–165]   → Test [165–180]
Window 3: Train [0–180]   → Test [180–195]
Window 4: Train [0–195]   → Test [195–200]
```

> **Không bao giờ có data leakage** — tập test không chứa bất kỳ điểm dữ liệu nào được dùng trong train.

### 3.2 Multi-Seed Averaging

Với mỗi window, chạy **3 random seeds** (42, 52, 62) cho các mô hình deep learning để ổn định kết quả khởi tạo ngẫu nhiên:

- **Sklearn models (GBT, RF, SVR, GLM, DNN):** Chỉ chạy 1 lần (seed đầu tiên), sau đó copy kết quả cho 2 seed còn lại — do các mô hình này rất ổn định với seed.
- **Deep Learning models (LSTM, GRU, CNN-LSTM, GCN-only, GAT-LSTM):** Chạy đủ 3 seeds vì trọng số khởi tạo ngẫu nhiên ảnh hưởng đáng kể đến kết quả.

### 3.3 Tổng hợp kết quả

- Kết quả 3 seeds trong một window → lấy **trung bình số học** cho window đó
- Kết quả qua nhiều windows → tính **mean ± std** và **95% Confidence Interval** (dùng `ddof=1`, `±1.96·SE`)

---

## 4. Cấu Hình Các Mô Hình

### 4.1 Sklearn Baselines

|Mô hình|Cấu hình|Ghi chú|
|---|---|---|
|**GBT** (Gradient Boosting)|`n_estimators=100`|Wrapped bởi `MultiOutputRegressor`|
|**RF** (Random Forest)|`n_estimators=100, n_jobs=-1`|Wrapped bởi `MultiOutputRegressor`|
|**SVR**|`C=1.0`|Không có `random_state`; Wrapped bởi `MultiOutputRegressor`|
|**GLM**|`LinearRegression()`|Không có hyperparameter|
|**DNN** (MLPRegressor)|`max_iter=500, early_stopping=True`|Wrapped bởi `MultiOutputRegressor`|

**Input format cho Sklearn:** `[W, N, F]` → flatten thành `[N_total, W×F]` (tabular)

### 4.2 Deep Learning Baselines

**Input format:** `[N_total, W, F]` (sequence format, giữ chiều thời gian)

|Mô hình|Kiến trúc|Hidden Dim|Batch Size|Shuffle|Epochs|Loss|
|---|---|---|---|---|---|---|
|**LSTM**|1-layer LSTM + Linear head|128|16|False|40|MSELoss|
|**GRU**|1-layer GRU + Linear head|128|16|False|40|MSELoss|
|**CNN-LSTM**|Conv1D(channels=32) + LSTM(128) + Linear|128|16|False|40|MSELoss|
|**GCN-only**|GCNConv×2(64) + LSTM(64) + Linear|64|N/A*|N/A|40|MSELoss|

> * GCN-only xử lý từng sequence trong graph format, không dùng DataLoader thông thường.

**Optimizer:** Adam, `lr=0.005`  
**Gradient clipping:** `max_norm=1.0`

### 4.3 GAT-LSTM (Mô hình đề xuất)

|Tham số|Giá trị|
|---|---|
|Kiến trúc|`StockGAT_LSTM_Deep` (model_type="deep")|
|GNN hidden|64|
|LSTM hidden|128|
|Attention heads|2|
|Epochs|40|
|Optimizer|Adam, `lr=0.005`, `weight_decay=1e-4`|
|Loss|`nn.MSELoss()` _(hiện tại)_|
|Update strategy|Full-batch giả lập: cộng dồn loss toàn bộ train sequences, backward 1 lần/epoch|
|Gradient clipping|`max_norm=1.0`|

> **Lưu ý quan trọng:** Trong phiên bản benchmark này (`run_benchmark_dl_only`), GAT-LSTM được train bằng **MSELoss thuần** (không dùng CorrelationLoss). CorrelationLoss (`0.6·MSE + 0.2·Corr + 0.2·Risk`) được định nghĩa trong `src.trainer` nhưng đang được comment out, chưa được áp dụng.

---

## 5. Metric Đánh Giá

Tất cả mô hình đều được đánh giá bởi **cùng một bộ metric** trên tập test:

|Metric|Công thức|Ý nghĩa|
|---|---|---|
|`mae_interval`|`0.5 × (MAE_min + MAE_max)`|**Primary metric** — sai số tuyệt đối trung bình của khoảng dự báo|
|`mae_min`|MAE trên `min_return`|Độ chính xác dự đoán đáy|
|`mae_max`|MAE trên `max_return`|Độ chính xác dự đoán đỉnh|
|`mse_interval`|`0.5 × (MSE_min + MSE_max)`|Sai số bình phương trung bình|
|`mse_min`, `mse_max`|MSE từng nhánh|Diagnostic|

---

## 6. Đánh Giá Tính Công Bằng Của Benchmark

### 6.1 Điểm mạnh (đảm bảo công bằng)

**✅ Không có data leakage về thời gian**  
Expanding window đảm bảo tập test không bao giờ xuất hiện trong train. Điều này rất quan trọng với dữ liệu tài chính do tính chất non-stationary.

**✅ Metric đánh giá thống nhất**  
Mọi mô hình đều được đo bằng `MAE_interval` và `MSE_interval` trên cùng một tập test, không có ưu ái nào cho mô hình nào.

**✅ Gradient clipping nhất quán**  
Tất cả DL models đều dùng `max_norm=1.0`, tránh lợi thế do gradient scale khác nhau.

**✅ Cùng learning rate cho DL baselines**  
LSTM, GRU, CNN-LSTM, GCN-only và GAT-LSTM đều dùng `lr=0.005`.

**✅ Multi-seed averaging**  
3 seeds cho DL models kiểm soát variance do khởi tạo ngẫu nhiên.

**✅ Confidence interval 95%**  
Dùng `ddof=1` và `±1.96·SE` cho phép so sánh thống kê có ý nghĩa.

---

### 6.2 Điểm cần lưu ý (tiềm ẩn thiên lệch)

**⚠️ GAT-LSTM hiện dùng MSELoss — không phải CorrelationLoss**

Đây là vấn đề lớn nhất cần làm rõ. Toàn bộ phần code CorrelationLoss đang bị **comment out**:

```python
# criterion = CorrelationLoss(weight_mse=0.9, weight_corr=0.05, weight_penalty=0.05)
criterion = nn.MSELoss()
```

Điều này có nghĩa là trong benchmark hiện tại, GAT-LSTM và các DL baseline **đều dùng cùng một loss function (MSE)**. Do đó:

- Nếu GAT-LSTM thắng → lợi thế đến từ **kiến trúc GAT** (graph attention), không phải từ custom loss
- Nếu GAT-LSTM thua → cần kiểm tra lại cách GAT xử lý graph so với GCN-only

**⚠️ GAT-LSTM dùng chiến lược Full-Batch, các baseline dùng Mini-Batch**

GAT-LSTM cộng dồn loss toàn bộ train sequences rồi backward 1 lần/epoch (full-batch). Trong khi đó, LSTM/GRU/CNN-LSTM dùng mini-batch size=16. Điều này tạo ra sự không đồng nhất về:

- Tần suất cập nhật gradient (GAT-LSTM: 1 lần/epoch vs baseline: nhiều lần/epoch)
- Implicit regularization: mini-batch SGD có noise gradient giúp tổng quát hóa tốt hơn so với full-batch GD

Điều này có thể làm GAT-LSTM **chậm hội tụ** hoặc **overfit** khác đi so với baseline.


**⚠️ GCN-only không có batch training**

GCN-only không dùng DataLoader hay mini-batch, mà duyệt từng sequence một trong graph format. Điều này khác về mặt optimization dynamics so với các baseline sequence model dùng batch=16.

---



![[Pasted image 20260331134917.png]]