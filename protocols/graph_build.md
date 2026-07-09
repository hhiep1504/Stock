1. Kiến trúc Dual-Path GCN (Ví dụ: Mô hình GSL-GNN)
Đối với mạng GCN, nếu bạn gộp trực tiếp đồ thị tĩnh (Sector) và đồ thị động (Correlation) thành một ma trận duy nhất, đồ thị sẽ trở nên quá đặc (dense), dẫn đến hiện tượng Over-smoothing (làm mượt quá mức) khiến mọi cổ phiếu bị cào bằng đặc trưng.
Kiến trúc Dual-Path GCN giải quyết việc này bằng cách tách ra chạy 2 luồng GCN song song:
Luồng 1 (Original/Static Graph): Xử lý ma trận kề gốc (ví dụ: đồ thị nhóm ngành tĩnh), cung cấp kiến thức nền tảng vững chắc và ổn định
,
.
Luồng 2 (Learned/Dynamic Graph): Xử lý trên ma trận kề động được mô hình tự học hoặc tính toán theo thời gian thực (ví dụ: Pearson Correlation Top-K) để bắt các tín hiệu hành vi ẩn
,
.
Cơ chế Fusion (Hợp nhất): Sau khi mỗi luồng GCN trích xuất xong đặc trưng, kết quả sẽ được cộng gộp (hoặc trung bình) dựa trên một trọng số điều chỉnh α (thường thiết lập α=0.5): 
H 
final
​
 =αH 
learned
​
 +(1−α)H 
original
​
 
.
Điểm mạnh: Cơ chế này đóng vai trò như một bộ điều chuẩn (regularization). Ngay cả khi đồ thị động (Learned Graph) chứa nhiều nhiễu trong các epoch đầu, đồ thị gốc (Original Graph) vẫn cung cấp một luồng truyền gradient (gradient pathway) cực kỳ ổn định, giúp GCN không bị sụp đổ
,
.
2. Kiến trúc Dual-Path GAT (Ví dụ: Dual-Graph BiLSTM-GAT và Heterogeneous GAT)
Kiến trúc GAT khi áp dụng Dual-Path linh hoạt và mạnh mẽ hơn rất nhiều nhờ cơ chế tự chú ý (Self-Attention).
Cơ chế Dual-Graph cơ bản: Theo nghiên cứu của Lu et al. (2025) về mô hình BiLSTM-GAT dự báo chứng khoán, kiến trúc sử dụng một cấu trúc đồ thị kép: một đồ thị ghi lại sự tương đồng kỹ thuật trong chuỗi thời gian (dynamic) và một đồ thị mã hóa mối quan hệ ngành cơ bản (static)
.
Xử lý bằng Attention: Thay vì trộn ma trận như GCN, kiến trúc này duy trì 2 kênh đồ thị độc lập. GAT sẽ thực hiện việc phân bổ sự chú ý trên từng kênh đồ thị một cách riêng biệt. Sau đó, mô hình sử dụng thêm một cơ chế attention phụ (additional attention mechanism) để quyết định xem ở thời điểm hiện tại, nên "tin tưởng" (fuse) vào đồ thị kỹ thuật hay đồ thị ngành nhiều hơn
.
Mở rộng thành Heterogeneous GAT: Bài báo của Tianhui Huang mở rộng Dual-Path thành Multi-Path. Bằng cách sử dụng module HeteroConv, mô hình gán các Attention Heads (Đầu chú ý) hoàn toàn tách biệt cho từng loại cạnh (ví dụ: Head 1 cho cạnh Correlation, Head 2 cho cạnh Sector)
. Điều này cho phép mạng GAT học các chiến lược gộp thông tin hoàn toàn khác nhau cho từng loại đồ thị mà không bị pha loãng sự chú ý
,
.
💡 Liên hệ trực tiếp với Project của bạn
Trong những phiên trao đổi trước, bạn đang xây dựng edge_index bao gồm cả Static Graph (Sector) và Dynamic Graph (Pearson Top-K).
Nếu dùng GCN-LSTM: Việc bạn cộng thẳng 2 đồ thị vào một edge_index có thể là nguyên nhân khiến model bị nhiễu do GCN không thể phân biệt đâu là cạnh tĩnh, đâu là cạnh động. Nếu muốn GCN tối ưu, bạn phải code theo dạng Dual-Path GCN (chia 2 nhánh GCN riêng rồi + lại)
.
Nếu dùng GAT-LSTM: Cơ chế Dual-Path / Heterogeneous chính là chìa khóa để GAT tỏa sáng. Bạn nên cho 2-4 Attention Heads chỉ chạy trên edge_index của Dynamic Graph, và 2-4 Attention Heads khác chỉ chạy trên edge_index của Static Graph, sau đó nối (concatenate) lại
,
. Cách làm này sẽ bảo vệ model của bạn triệt để trước mọi sự tấn công (Over-smoothing, Attention Dilution)