Dưới đây là phần mô tả **chi tiết cho hai lần tune: `baseline4` và `baseline_plus_sma12`.

**Experiment Setup**
Hai bộ đặc trưng được tune riêng biệt để đảm bảo so sánh công bằng. Cả hai dùng cùng dữ liệu, cùng cách chia time-series, cùng số lượng Optuna trials, cùng metric, cùng graph setting và cùng tập model. Điểm khác duy nhất là số lượng input features.

Config dùng:
- [tune_baseline4.json](../configs/tune_baseline4.json)
- [tune_baseline_plus_sma12.json](../configs/tune_baseline_plus_sma12.json)

**Dữ liệu**
Dữ liệu giá ngày lấy từ:

```text
dataset/stock_market_19_24.csv
```

Trong workspace hiện tại, dữ liệu chạy từ:

```text
2019-06-28 đến 2024-12-31
```

Gồm `26` cổ phiếu HOSE. Dữ liệu ngày được aggregate sang tuần ISO bằng `Year_Week`, ví dụ `2019_W26`, `2024_W52`, `2025_W01`.

Lưu ý: `2025_W01` không có nghĩa là dùng dữ liệu năm 2025. Nó là ISO-week label cho các ngày cuối tháng 12/2024, cụ thể Dec 30-31, 2024 nằm trong ISO week 1 của năm 2025.

**Feature Sets**
`baseline4` gồm 4 feature gốc:

```text
f_std
f_mean
f_return
f_skew
```

`baseline_plus_sma12` gồm 5 feature:

```text
f_std
f_mean
f_return
f_skew
sma_ratio_w12
```

Trong đó:

```text
sma_ratio_w12 = weekly_close / SMA_12(weekly_close) - 1
```

Vì `sma_ratio_w12` cần rolling window 12 tuần, nên 11 tuần đầu bị loại khỏi feature set này.

**Period And Sequence Construction**
Tất cả model dùng `window_size = 8`.

Nghĩa là mỗi sample dùng 8 tuần liên tiếp làm input để dự đoán min-return và max-return của tuần tiếp theo:

```text
[X_t, X_{t+1}, ..., X_{t+7}] -> y_{t+8}
```

Ví dụ:

```text
baseline4:
2019_W26 ... 2019_W33 -> predict 2019_W34

baseline_plus_sma12:
2019_W37 ... 2019_W44 -> predict 2019_W45
```

Số lượng period/sequence hiện tại:

```text
baseline4:
valid weekly periods = 288
period range = 2019_W26 to 2025_W01
sequences = 288 - 8 = 280
Optuna tuning sequences = 276
hold-out sequences = 4

baseline_plus_sma12:
valid weekly periods = 277
period range = 2019_W37 to 2025_W01
sequences = 277 - 8 = 269
Optuna tuning sequences = 265
hold-out sequences = 4
```

`split_idx = -4`, nên 4 sequence cuối không được dùng trong Optuna tuning. Chúng được giữ lại như hold-out tail.

**Validation Protocol**
Optuna tuning chỉ chạy trên training portion, tức là toàn bộ sequence trừ 4 sequence cuối.

Sau đó training portion được chia bằng:

```text
TimeSeriesSplit(n_splits=5)
```

Đây là chronological split, không shuffle. Mỗi fold dùng dữ liệu quá khứ để validate trên đoạn thời gian sau đó. Cách này tránh leakage so với random split.

Scaler cũng được fit lại trên training prefix của từng fold, không fit trên toàn bộ dữ liệu. Nói cách khác, normalization không nhìn thấy validation/test future periods.

**Graph Construction**
Cả hai feature set dùng cùng graph setting:

```text
graph_mode = hybrid
similarity_metric = pearson
top_k = 4
corr_threshold = 0.7
use_static_graph = true
use_arm = false
```

Hybrid graph nghĩa là graph cuối cùng được tạo bằng cách merge:

```text
static graph + dynamic graph
```

Static graph dựa trên sector/domain relation. Dynamic graph được tính theo Pearson correlation trong chính 8-week input window của từng sample.

**Optuna Tuning**
Mỗi model family được tune độc lập với:

```text
n_trials = 100
direction = minimize
metric = mae_interval
max_epochs_per_trial = 10
sampler_seed = 42
loss = mse
```

Metric tối ưu là:

```text
mae_interval
```

tức MAE trung bình trên hai output:

```text
min return prediction
max return prediction
```

Các model được tune:

```text
LSTM
GRU
CNN-LSTM
Temporal GCN
GAT-LSTM
```

Vì có 5 model và 100 trials/model:

```text
5 x 100 = 500 Optuna trials cho mỗi feature set
```

Tổng cộng cho `baseline4` và `baseline_plus_sma12`:

```text
1000 Optuna trials
```


> Hyperparameter tuning was conducted separately for the original four-feature baseline and the SMA-augmented feature set. Daily stock prices from 26 HOSE stocks were aggregated into ISO-weekly observations. For each sample, an 8-week historical window was used to predict the next-week minimum and maximum return interval. The original baseline used four weekly features: standard deviation, mean price, return, and skewness. The SMA-augmented setting added a 12-week SMA ratio, defined as weekly close divided by its 12-week moving average minus one. Because this feature requires a 12-week rolling history, the first 11 weekly periods were excluded from the augmented setting.  
>
> For model selection, the final four chronological sequence samples were held out, and the remaining sequences were used for Optuna tuning with five-fold TimeSeriesSplit validation. No random shuffling was applied. Feature scaling was fitted only on the training prefix within each fold to avoid temporal leakage. Each model family was tuned independently for 100 Optuna trials using MAE over the predicted return interval as the optimization objective. The same tuning protocol was applied to LSTM, GRU, CNN-LSTM, Temporal GCN, and GAT-LSTM under both feature settings.



# Tại sao lại là sma12

**Vì Sao Chọn SMA12**
Ban đầu, mô hình chỉ dùng `baseline4`:

```text
f_std
f_mean
f_return
f_skew
```

Sau đó mình mở rộng không gian feature bằng nhiều technical indicators và nhiều window khác nhau, chứ không chọn SMA12 thủ công ngay từ đầu. Các nhóm feature được thử gồm:

```text
SMA ratio
EMA ratio
Bollinger width
Bollinger percent-b
RSI
MACD
Hurst exponent
Entropy
Relative position
Alpha excess
Weekly range lag
```

Với mỗi nhóm, mình test nhiều window/parameter. Ví dụ:

```text
SMA ratio: 2, 3, 4, 5, 6, 8, 12 tuần
EMA ratio: 2, 3, 4, 5, 6, 8, 12 tuần
RSI: 4, 5, 8, 14, 20 tuần
Bollinger: 4, 5, 6, 8, 12 tuần, k = 1.5 / 2.0
MACD: 3/6, 4/8, 5/10, 6/12, 8/17, 12/26
Hurst/Entropy: 12, 20, 26 tuần
```

Tổng cộng có `60` candidate feature variants được sweep.

Mục tiêu của bước này không phải là train full benchmark ngay, mà là **screening**: kiểm tra từng feature khi thêm riêng lẻ vào `baseline4` có giúp giảm MAE hay không.

Kết quả sweep cho thấy `sma_ratio_w12` là một trong những feature đơn lẻ tốt nhất, và cụ thể là đứng đầu theo improvement trong single-feature screening:

```text
sma_ratio_w12:
MAE = 0.020604
baseline same-row MAE = 0.023897
delta = -0.003294
```

Nói đơn giản:

```text
baseline4 + sma_ratio_w12
```

cho kết quả tốt hơn baseline tương ứng trong vòng screening ban đầu.

**Vì Sao Không Giữ Tất Cả Feature**
Sau bước sweep, mình cũng không kết luận vội là cứ thêm càng nhiều feature càng tốt. Dataset nhỏ, nên nếu đưa quá nhiều technical indicators vào deep learning model thì rất dễ bị:

```text
overfitting
multicollinearity
loss of early periods do rolling windows
curse of dimensionality
```

Mình đã thử thêm các bộ feature rộng hơn, ví dụ:

```text
screened_liquid_top5
screened_with_hurst
baseline_plus_hurst26
baseline_plus_sma12
```

Sau đó chạy common-period benchmark để so công bằng về thời gian. Kết quả cho thấy các bộ feature rộng như `screened_with_hurst` hoặc `screened_liquid_top5` không cải thiện ổn định. Trong khi đó `baseline_plus_sma12` là bộ mở rộng nhỏ nhất, đơn giản nhất, và từng có tín hiệu tốt nhất ở bước screening.

Vì vậy, thay vì so sánh quá nhiều feature set phức tạp, mình rút về câu hỏi nghiên cứu gọn hơn:

```text
Liệu chỉ thêm feature kỹ thuật mạnh nhất từ screening, sma_ratio_w12,
có cải thiện mô hình so với baseline4 sau khi retune công bằng không?
```

**Vì Sao Cuối Cùng Chỉ So Với baseline_plus_sma12**
Lý do là để giữ experiment sạch và có thể viết báo cáo rõ ràng:

```text
baseline4
```

là mô hình gốc.

```text
baseline_plus_sma12
```

là phiên bản mở rộng tối giản, chỉ thêm feature tốt nhất từ screening.

So sánh này kiểm tra đúng contribution của một feature mới mà không làm nhiễu bởi việc thêm quá nhiều indicators cùng lúc.

Sau đó cả hai feature set đều được retune với cùng protocol:

```text
5 model families
100 Optuna trials/model
TimeSeriesSplit 5 folds
same metric: mae_interval
same graph setting
same window_size = 8
```

Kết quả retune cho thấy sau khi tuning công bằng, `baseline4` lại tốt hơn `baseline_plus_sma12` ở cả 5 model. Đây là kết quả rất quan trọng, vì nó chứng minh rằng:

```text
SMA12 có tín hiệu tốt ở bước screening,
nhưng không tạo ra cải thiện bền vững sau full hyperparameter tuning.
```

**Câu Viết Cho Báo Cáo**
Bạn có thể viết như này:

> The SMA12 ratio was not selected arbitrarily. Before the final comparison, a feature screening stage was conducted over 60 candidate technical indicators, including multiple variants of moving averages, Bollinger indicators, RSI, MACD, Hurst exponent, entropy, relative position, alpha excess, and lagged weekly range. Each candidate was evaluated by adding it individually to the original four-feature baseline. Among these candidates, the 12-week SMA ratio produced the strongest single-feature improvement, reducing MAE from 0.023897 to 0.020604 in the screening experiment.  
>
> However, because the dataset is relatively small, adding many technical indicators simultaneously may increase dimensionality, reduce the available valid period due to rolling-window requirements, and increase overfitting risk. Therefore, broader feature sets such as screened multi-indicator combinations were evaluated but did not show stable improvement under common-period benchmarking. The final feature comparison was thus simplified to a controlled test between the original baseline and the minimal SMA-augmented feature set. This design isolates the effect of the strongest screened feature while keeping the input dimensionality low.  
>
> Both the original baseline and the SMA12-augmented feature set were then retuned under the same Optuna protocol. This final retuning stage showed that although SMA12 was promising during single-feature screening, its advantage did not persist after model-level hyperparameter optimization. This suggests that the information captured by SMA12 may overlap with the original return and volatility features, or that the additional rolling indicator introduces noise relative to the size of the dataset.
