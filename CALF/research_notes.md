# CALF: Cross-modal Knowledge Distillation for Time Series Forecasting
*(Reference Guide and Research Notes)*

This document serves as a comprehensive reference guide for the **CALF** (**C**ross-modal Knowledge **A**lignment and **L**LM **F**orecasting) project, based on the research paper: *"Taming Pre-trained LLMs for Generalized Time Series Forecasting through Cross-modal Knowledge Distillation"* ([arXiv:2403.07300](https://arxiv.org/abs/2403.07300)).

---

## 1. Executive Summary & Core Concept

Standard methods for adapting Large Language Models (LLMs) to time series forecasting either fine-tune the LLM directly or train linear/MLP projection heads on top of frozen LLM blocks. **CALF** introduces a dual-branch cross-modal knowledge distillation framework:
*   **Teacher Branch (Text Modality)**: Aligns the time-series input into the textual semantic space of a pre-trained LLM using pre-calculated Word Token Embeddings (WTE) and cross-attention.
*   **Student Branch (Time Modality)**: Processes the normalized time-series input directly.
*   **Knowledge Distillation**: Aligns the student branch's intermediate feature maps and final prediction logits to match the teacher's, transferring rich, pre-trained semantic priors to the time-series model.

```
       [ Time Series Input: B x L x M ]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 ┌──────────────┐            ┌──────────────┐
 │ Time Student │            │ Text Teacher │
 │    Branch    │            │    Branch    │
 └──────┬───────┘            └──────┬───────┘
        │                           │
        ▼                           ▼
 ┌──────────────┐            ┌──────────────┐
 │ Frozen GPT-2 │            │ Frozen GPT-2 │
 │ (Time Modality)           │ (Text Modality)
 └──────┬───────┘            └──────┬───────┘
        │                           │
        ├── Feature Alignment ◄─────┤  (Feature Distillation Loss)
        │                           │
        ▼                           ▼
 ┌──────────────┐            ┌──────────────┐
 │ Task Head    │            │ Task Head    │
 └──────┬───────┘            └──────┬───────┘
        │                           │
        ├── Prediction Alignment ◄──┘  (Logits Distillation Loss)
        ▼
   [ Final Output ] ◄───────────────► [ Ground Truth ] (Task Loss)
```

---

## 2. Key Mathematical Formulations & Losses

CALF optimizes the student model using a multi-component distillation loss function:

$$\mathcal{L}_{\text{total}} = w_{\text{task}} \mathcal{L}_{\text{task}} + w_{\text{logits}} \mathcal{L}_{\text{logits}} + w_{\text{feature}} \mathcal{L}_{\text{feature}}$$

### 2.1 Task Loss ($\mathcal{L}_{\text{task}}$)
Measures the error between the student prediction ($\hat{\mathbf{Y}}_{\text{time}}$) and the ground truth target ($\mathbf{Y}$):
*   For **Long-term Forecasting**: L1 or MSE loss.
*   For **Short-term Forecasting**: SMAPE, MAPE, or MASE loss.

### 2.2 Logits Loss ($\mathcal{L}_{\text{logits}}$)
Constrains the student's forecasting outputs to mimic the outputs produced by the teacher:

$$\mathcal{L}_{\text{logits}} = \text{MSE}(\hat{\mathbf{Y}}_{\text{time}}, \hat{\mathbf{Y}}_{\text{text}})$$

### 2.3 Feature Loss ($\mathcal{L}_{\text{feature}}$)
Forces the intermediate states of the Time branch to align with the Text branch across all layers. It applies an exponential decay factor to prioritize the alignment of earlier layers:

$$\mathcal{L}_{\text{feature}} = \sum_{i=1}^{N_{\text{layers}}} (0.8)^{N_{\text{layers}} - i} \cdot \text{DistillLoss}(\mathbf{H}_{\text{time}}^{(i)}, \mathbf{H}_{\text{text}}^{(i)})$$

*Where:*
*   $\mathbf{H}_{\text{time}}^{(i)}$ and $\mathbf{H}_{\text{text}}^{(i)}$ are intermediate representations at layer $i$.
*   Supported distillation functions include standard L1/MSE, **MMD (Maximum Mean Discrepancy)**, **CORAL**, or contrastive semantic loss.

---

## 3. High-Level Directory Structure

```
CALF/
├── datasets/                 # Place datasets here (ETT, electricity, traffic, weather, m4)
├── models/                   # Neural network architectures
│   ├── GPT4TS.py             # Main CALF model with Dual-Branch GPT-2
│   ├── GPT2_arch.py          # Customized GPT-2 wrapper to output intermediate states
│   └── Embed.py              # Temporal, positional, and token embedding layers
├── exp/                      # Experiment orchestrators
│   ├── exp_basic.py          # Base experiment class
│   ├── exp_long_term_forecasting.py   # Long-term forecasting loop (train/test/bootstrap)
│   └── exp_short_term_forecasting.py  # Short-term forecasting loop
├── utils/                    # Helper utilities
│   ├── distillationLoss.py   # Calculates combined loss functions
│   ├── ditill_utils.py       # Computes MMD, CORAL, and Contrastive Semantic Losses
│   └── metrics.py            # Accuracy metrics (MAE, MSE, RMSE, MAPE)
├── scripts/                  # Shell scripts for executing experiments
│   └── long_term_forecasting/
│       ├── ETTh_GPT2.sh      # Shell script for ETTh datasets
│       └── electricity.sh    # Shell script for Electricity dataset
├── run.py                    # Main entrypoint parsing CLI parameters and launching tasks
├── pca.py                    # Extracts primary components of Word Token Embeddings
└── wte_pca_500.pt            # Extracted word token embeddings (500 principal components)
```

---

## 4. In-Depth Component Analysis

### 4.1 Entrypoint: `run.py`
Parses options for different forecasting datasets, shapes, loss functions, GPU configs, and model setups.
*   **Key Parameters**:
    *   `--task_name`: Task to run (`long_term_forecast`, `short_term_forecast`, `imputation`, `classification`, `anomaly_detection`).
    *   `--model_id`: Determines experiment run/variant (`ori`, `dropAttn_keepWE`, `llm_to_attn`, `llm_to_trsf`, `randomInit`).
    *   `--seq_len` / `--pred_len`: Input and prediction horizon lengths.
    *   `--task_w`, `--logits_w`, `--feature_w`: Weighted multipliers for combined losses.
    *   `--gpt_layers`: Number of layers to extract from GPT-2 (default: 32).
    *   `--word_embedding_path`: Path to WTE PCA artifact (default: `"wte_pca_500.pt"`).

### 4.2 Main Architecture: `models/GPT4TS.py`
Constructs the model `Model(configs, device)`.
*   Loads **two identical pre-trained GPT-2 models** (`gpt2` for Time, `gpt2_text` for Text).
*   Freezes the majority of parameters to leverage pre-trained language knowledge, only keeping LayerNorms, Word Positional Embeddings (`wpe`), or custom LoRA/projection parameters trainable:
    ```python
    for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        if 'ln' in name or 'wpe' in name or 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    ```
*   **Encoder PCA (`Encoder_PCA`)**: Maps the time-series tokens (`B, C, L`) into the hidden dimension ($768$), projects it, feeds it to a transformer encoder, and executes multi-head cross-attention against the text word token embeddings (`word_embedding`).
*   **Decoder**: A standard linear layer mapping the outputs back to the forecasting horizon (`configs.pred_len`).

### 4.3 Loss Handler: `utils/distillationLoss.py`
Calculates intermediate feature loss, output logits loss, and task loss:
```python
# 1. Feature Loss (Intermediate Representations)
if intermidiate_feat_time is not None:
    feature_loss = sum([
        (0.8**idx) * self.feature_loss(feat_time, feat_text)
        for idx, (feat_time, feat_text) in enumerate(
            zip(intermidiate_feat_time[::-1], intermidiate_feat_text[::-1])
        )
    ])

# 2. Logits Loss (Teacher-Student output alignment)
if outputs_text is not None:
    logits_loss = self.logits_loss(outputs_time, outputs_text)

# 3. Task Loss (Ground Truth Comparison)
task_loss = self.task_loss(outputs_time, batch_y)

# Total Loss
total_loss = self.task_w * task_loss + self.logits_w * logits_loss + self.feature_w * feature_loss
```

---

## 5. Experiment Execution Pipeline

To reproduce or adapt experiments, the following steps are performed:

### Step 1: Initialize Word Token Embeddings (WTE)
Extract and save the principal components of the word token embeddings:
```powershell
python pca.py
```
This produces `wte_pca_500.pt` in the project root.

### Step 2: Configure and Execute Scripts
Modify parameters in scripts within `./scripts/long_term_forecasting/`. For example, `ETTh_GPT2.sh` might run:
```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_96_96_ori \
  --model GPT4TS \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --features M \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 768 \
  --gpt_layers 6 \
  --train_epochs 10 \
  --batch_size 256 \
  --learning_rate 0.0001 \
  --log_fine_name ETTh1_96_96_ori.txt
```

### Step 3: Analyze Evaluation & Benchmarks
*   **Checkpoints**: Saved under `./checkpoints/`.
*   **Metrics**: Logged inside the file specified by `--log_fine_name`.
*   **NPY Outputs**: Predicted vs. true target arrays saved inside `./results/` for plotting.

---

## 6. Experimental Results & Benchmarks (BTCUSDT 15-Minute Dataset)

The following tables summarize the experimental results of the CALF model on the high-frequency **BTCUSDT 15-minute trading dataset** across various lookback/forecast horizons ($96 \to \{16, 32, 96\}$). Four model variants were evaluated:
1.  **`ori` (Original)**: Standard dual-branch CALF cross-modal alignment architecture.
2.  **`dropAttn_keepWE`**: Drops cross-attention maps but retains the text Word Token Embeddings (WTE).
3.  **`llm_to_attn`**: Distills language representation priors into a customized temporal cross-attention mechanism.
4.  **`llm_to_trsf`**: Distills language representation priors into a custom transformer structure.

### 6.1 Unified Benchmark Results Table

| Prediction Length (`pred_len`) | Model Variant | Mean Absolute Error (MAE) | Mean Squared Error (MSE) |
| :--- | :--- | :--- | :--- |
| **16 Steps** (4 hours) | `ori` (Original) | $0.16842 \pm 0.000415$ | $0.26695 \pm 0.000104$ |
| | `dropAttn_keepWE` | $0.16959 \pm 0.000344$ | **$0.26188 \pm 0.000509$** |
| | `llm_to_attn` | $0.17056 \pm 0.000284$ | $0.26546 \pm 0.001537$ |
| | `llm_to_trsf` | $0.17067 \pm 0.000970$ | $0.26732 \pm 0.002104$ |
| **32 Steps** (8 hours) | `ori` (Original) | $0.18722 \pm 0.002459$ | $0.29046 \pm 0.003274$ |
| | `dropAttn_keepWE` | **$0.18644 \pm 0.001557$** | **$0.28703 \pm 0.002701$** |
| | `llm_to_attn` | $0.18650 \pm 0.000162$ | $0.28817 \pm 0.000123$ |
| | `llm_to_trsf` | $0.18660 \pm 0.000783$ | $0.28873 \pm 0.000664$ |
| **96 Steps** (24 hours) | `ori` (Original) | $0.21606 \pm 0.000641$ | $0.32962 \pm 0.001445$ |
| | `dropAttn_keepWE` | $0.21529 \pm 0.000337$ | **$0.32634 \pm 0.000307$** |
| | `llm_to_attn` | $0.21689 \pm 0.000663$ | $0.32765 \pm 0.000207$ |
| | `llm_to_trsf` | **$0.21467 \pm 0.000354$** | $0.32785 \pm 0.000135$ |

### 6.2 Key Observations

*   **Horizon Impact**: As the forecast horizon increases from $16 \to 96$, both MAE and MSE naturally increase across all models due to the rising uncertainty of longer-term high-frequency forecasting.
*   **Ablation performance**: Interestingly, `dropAttn_keepWE` demonstrates exceptional robustness, achieving the lowest MSE in 16-step ($0.26188$) and 32-step ($0.28703$) predictions. This suggests that simply preserving Word Token Embeddings (WTE) without the complexity of cross-attention maps can act as an effective regularizer.
*   **Distillation stability**: The `llm_to_attn` and `llm_to_trsf` models maintain highly stable standard deviations (often lower than the original model), proving that cross-modal distillation from frozen LLMs transfers robust, low-variance forecasting capability.

---
*Created as a permanent research documentation guide for developers and AI assistants pair-programming on CALF.*
