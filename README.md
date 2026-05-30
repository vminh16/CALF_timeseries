# CALF BTCUSDT Forecasting Lab

Repository này ghi lại quá trình tôi tái lập, điều chỉnh và đánh giá CALF trên dữ liệu
`BTCUSDT` khung thời gian 15 phút. Phạm vi làm việc hiện tại chỉ tập trung vào
`CALF/`; các mô hình khác từ workspace ban đầu không còn thuộc phạm vi commit.

## Tham Chiếu Báo Cáo

Nền tảng của repo là báo cáo:

> Peiyuan Liu et al., **CALF: Aligning LLMs for Time Series Forecasting via
> Cross-modal Fine-Tuning**, arXiv:2403.07300.

Báo cáo mô tả CALF cho multivariate time-series forecasting (MTSF): một temporal
target branch nhận chuỗi thời gian, một textual source branch nhận biểu diễn văn
bản đã căn chỉnh, cùng feature regularization loss và output consistency loss để
giảm khoảng cách giữa hai modality.

Paper: [arXiv:2403.07300](https://arxiv.org/abs/2403.07300)

## Mục Tiêu Thực Nghiệm

Tôi sử dụng dữ liệu `BTCUSDT_15m_calf.csv` với:

- Input window: `seq_len=96`, tương đương 24 giờ.
- Forecast horizons: `pred_len={16, 32, 96}`, tương đương `{4, 8, 24}` giờ.
- Features: `open`, `high`, `low`, `volume`, `quote_volume`,
  `number_of_trades`, `close`.
- Ablations: `ori`, `dropAttn_keepWE`, `llm_to_attn`, `llm_to_trsf`.
- Model dimension: `d_model=768`, `d_ff=768`, `gpt_layers=6`.

### Lưu Ý Về Target

Cấu hình hiện tại dùng `--features M --target close`. Trong implementation CALF,
`target=close` đưa `close` xuống cột cuối, nhưng `data_x`, `data_y`, output và loss
vẫn chứa toàn bộ 7 features. Vì vậy đây là bài toán multivariate-to-multivariate,
không phải bài toán chỉ tối ưu riêng `close`.

Nếu mục tiêu tiếp theo là dùng toàn bộ feature quá khứ để dự đoán riêng `close`,
cần thiết kế lại objective theo hướng `MS` hoặc custom close-only loss/output và
train checkpoint mới.

## Những Phần Đã Thực Hiện

### Tương Thích Với Môi Trường Local GPU

- Chạy CALF trên Windows với Conda env `llm-ts`.
- Xác nhận GPU local hoạt động với CUDA-enabled PyTorch.
- Thêm loader an toàn cho NumPy-backed artifact `wte_pca_500.pt` trên PyTorch
  `2.6+`, vẫn giữ `weights_only=True`.
- Dùng loader state dict an toàn cho checkpoint model.
- Bổ sung `CALF/requirements-modern.txt` cho dependency hiện đại.

### Training Và Evaluation

- Thêm script train local:
  `CALF/scripts/long_term_forecasting/run_btc_calf_local.bat`.
- Thêm script eval checkpoint `pred_len=16`:
  `CALF/scripts/long_term_forecasting/run_btc_calf_eval_local.bat`.
- Khôi phục nhánh `--is_training 0` để load đúng
  `checkpoints/<setting>/checkpoint.pth`.
- Lưu mỗi lần eval vào thư mục riêng để không overwrite:

```text
CALF/results/<setting>/eval_YYYYMMDD_HHMMSS_itr0/
  metrics_0.npy
  pred_0.npy
  true_0.npy
  run_info.txt
```

### Notebook Phân Tích

Notebook:
`CALF/notebooks/btc_pred16_checkpoint_evaluation.ipynb`

Notebook không chạy inference lại. Nó đọc artifact eval, rebuild CALF test
dataloader, kiểm tra `true_0.npy` khớp thứ tự `batch_y`, kiểm tra lại metrics, rồi
visualize 96 bước history và 16 bước forecast trong đúng CALF eval/scaled space.

Notebook có cell riêng cho từng ablation và bảng so sánh persistence baseline.

## Kết Quả Training BTCUSDT

Các metric dưới đây được ghi trong `CALF/BTCUSDT_15m_result.txt`. Đây là metric
aggregate trên toàn bộ 7 features.

| Pred len | Variant | MAE | MSE |
| --- | --- | ---: | ---: |
| 16 | `ori` | 0.16842 +/- 0.000415 | 0.26695 +/- 0.000104 |
| 16 | `dropAttn_keepWE` | 0.16959 +/- 0.000344 | **0.26188 +/- 0.000509** |
| 16 | `llm_to_attn` | 0.17056 +/- 0.000284 | 0.26546 +/- 0.001537 |
| 16 | `llm_to_trsf` | 0.17067 +/- 0.000970 | 0.26732 +/- 0.002104 |
| 32 | `ori` | 0.18722 +/- 0.002459 | 0.29046 +/- 0.003274 |
| 32 | `dropAttn_keepWE` | **0.18644 +/- 0.001557** | **0.28703 +/- 0.002701** |
| 32 | `llm_to_attn` | 0.18650 +/- 0.000162 | 0.28817 +/- 0.000123 |
| 32 | `llm_to_trsf` | 0.18660 +/- 0.000783 | 0.28873 +/- 0.000664 |
| 96 | `ori` | 0.21606 +/- 0.000641 | 0.32962 +/- 0.001445 |
| 96 | `dropAttn_keepWE` | 0.21529 +/- 0.000337 | **0.32634 +/- 0.000307** |
| 96 | `llm_to_attn` | 0.21689 +/- 0.000663 | 0.32765 +/- 0.000207 |
| 96 | `llm_to_trsf` | **0.21467 +/- 0.000354** | 0.32785 +/- 0.000135 |

## Kết Quả Eval Checkpoint `pred_len=16`

Tôi đã load lại checkpoint `itr=0` và eval trên test split. Mỗi artifact có shape
`(13952, 16, 7)`.

| Variant | MAE | MSE |
| --- | ---: | ---: |
| `ori` | **0.168834** | 0.267050 |
| `dropAttn_keepWE` | 0.169939 | **0.262389** |
| `llm_to_attn` | 0.170279 | 0.266996 |
| `llm_to_trsf` | 0.171636 | 0.269421 |

Notebook đã kiểm tra:

- `true_0.npy` khớp ground truth từ CALF dataloader: max absolute diff `0`.
- MAE/MSE tính lại từ `pred_0.npy` và `true_0.npy` khớp `metrics_0.npy`: diff `0`.

## Quan Sát Riêng Với `close`

Aggregate metric trên 7 features không phản ánh đầy đủ chất lượng dự đoán
`close`. Khi so với persistence baseline, tức lặp lại `close` cuối cùng của input
cho toàn bộ 16 bước tương lai:

| Variant | Close MAE | So với persistence |
| --- | ---: | ---: |
| Persistence baseline | **0.023059** | 0.00% |
| `ori` | 0.028175 | +22.18% |
| `dropAttn_keepWE` | 0.025953 | +12.55% |
| `llm_to_attn` | 0.026581 | +15.27% |
| `llm_to_trsf` | 0.028461 | +23.42% |

Forecast `close` có xu hướng đi ngang: với `ori`, mean absolute step size của
forecast là `0.002199`, trong khi ground truth là `0.008343`. Đây là hành vi thật
trong artifact eval, không phải lỗi alignment của notebook.

Kết quả aggregate vẫn tương đối tốt vì model cải thiện so với persistence trên
`volume`, `quote_volume` và `number_of_trades`, trong khi kém hơn trên nhóm
`open`, `high`, `low`, `close`.

## Cách Chạy

### Chuẩn Bị Env

```bat
conda activate llm-ts
cd /d "C:\Users\USER\Desktop\time_series\LLM for timeseries\LLMsForTimeSeries\CALF"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

PyTorch CUDA cần được cài riêng. Các dependency còn lại được mô tả trong:

```text
CALF/requirements-modern.txt
```

### Train

```bat
scripts\long_term_forecasting\run_btc_calf_local.bat
```

### Eval Checkpoint `pred_len=16`

```bat
scripts\long_term_forecasting\run_btc_calf_eval_local.bat
```

Script eval hiện dùng checkpoint `itr=0`, tương ứng nhánh eval trong
`CALF/run.py`.

### Visualize

Mở notebook:

```text
CALF/notebooks/btc_pred16_checkpoint_evaluation.ipynb
```

Sau đó chạy các cell từ trên xuống dưới.

## Cấu Trúc Repo

```text
CALF/
  data_provider/       # Dataset loaders
  exp/                 # Train/eval loops
  models/              # CALF model và ablations
  notebooks/           # Artifact validation và visualization
  scripts/             # Train/eval commands
  utils/               # Metrics, losses, compatibility helpers
docs/context/          # Ghi chú phân tích source
tests/                 # Compatibility tests cho CALF
```

Dataset, checkpoint, logs và artifact eval được giữ local và không commit.

## Trạng Thái Nghiên Cứu

Repo hiện tái lập được CALF cho BTCUSDT multivariate forecasting và có pipeline
eval/visualize checkpoint có kiểm tra alignment. Bước tiếp theo hợp lý là tạo
thực nghiệm close-only objective, train checkpoint mới, rồi so sánh với
persistence baseline trên riêng target `close`.

