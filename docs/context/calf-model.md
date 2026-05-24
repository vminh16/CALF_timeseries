# CALF Model Context

Last verified: 2026-05-23 from local repository files and compatibility checks.

## What CALF Is In This Repo

`CALF/` is the local implementation of "Taming Pre-trained LLMs for Generalized Time Series Forecasting through Cross-modal Knowledge Distillation." In this repo it is also used as an ablation target for testing whether the LLM component matters for time-series forecasting.

## Execution Flow

1. `CALF/run.py` parses CLI args and chooses an experiment class by `--task_name`: `long_term_forecast`, `short_term_forecast`, `imputation`, `classification`, or `anomaly_detection`.
2. Each `CALF/exp/exp_*.py` class builds the model through `Exp_Basic.model_dict`.
3. `Exp_Basic.model_dict` currently maps only `"GPT4TS"` to `CALF/models/GPT4TS.py`.
4. Training loops call `self.model(batch_x)` and expect a dict with `outputs_time`, `outputs_text`, `intermidiate_time`, and `intermidiate_text`.
5. For evaluation, the task prediction is taken from `outputs_time`.

## Main Architecture

`CALF/models/GPT4TS.py` defines the active model:

- `Encoder_PCA` maps each variable's input history from `seq_len` to `d_model`, applies a small Transformer encoder, then cross-attends the time representation to PCA word-token embeddings loaded from `--word_embedding_path` (`wte_pca_500.pt` by default).
- The main model creates two GPT-2 branches with `AccustumGPT2Model.from_pretrained('gpt2')`: `gpt2` for the time path and `gpt2_text` for the text/word-embedding path.
- Both GPT-2 branches are truncated to `configs.gpt_layers`.
- The default trainable GPT-2 parameters are limited: `gpt2` trains names containing `ln`, `wpe`, or `lora`; `gpt2_text` trains names containing `wpe`; other GPT-2 parameters are frozen.
- `out_layer` is task-specific: forecast outputs `pred_len`, classification outputs `num_class`, imputation/anomaly outputs `seq_len`.
- `CALF/models/GPT2_arch.py` is now a thin wrapper over Hugging Face `GPT2Model.forward`; it forces `output_hidden_states=True` and `return_dict=True`, then returns `(last_hidden_state, hidden_states)` for the existing CALF callers.

## Distillation Loss

`CALF/utils/distillationLoss.py` combines three terms:

- `task_loss`: prediction vs. task label.
- `logits_loss`: `outputs_time` vs. `outputs_text` when a text branch output exists.
- `feature_loss`: weighted hidden-state alignment between time and text intermediate features.

The final loss is:

```text
task_w * task_loss + logits_w * logits_loss + feature_w * feature_loss
```

Defaults in `CALF/run.py` are `task_w=1.0`, `logits_w=1.0`, and `feature_w=0.01`.

## Ablation Modes

The model behavior is controlled by substring checks inside `--model_id`:

| Substring in `model_id` | Behavior |
| --- | --- |
| `ori` | Uses both GPT-2 branches and distillation outputs. |
| `dropAttn_keepWE` | Drops GPT-2 branches and keeps the encoder/word-embedding alignment path. |
| `llm_to_attn` | Replaces the time LLM path with local multi-head attention. |
| `llm_to_trsf` | Replaces the time LLM path with a local Transformer encoder. |
| `randomInit` | Replaces pretrained GPT-2 with randomly initialized `GPT2Model(GPT2Config())`. |

If a forecasting `model_id` does not include one of the expected substrings, `forecast()` has no explicit fallback return. Keep ablation names consistent with the scripts.

## Data Shape Notes

For long-term forecasting, comments and code show `batch_x` as `[batch, seq_len, channels]`. The model normalizes over time, rearranges to `[batch, channels, seq_len]`, treats channels as tokens, then maps back to `[batch, pred_len, channels]`.

Approximate long-forecast forward complexity from the visible modules:

- Time: `O(B * M * seq_len * d_model + B * M * K * d_model + B * L_gpt * M^2 * d_model)`, where `B` is batch size, `M` channels, `K` PCA word-embedding tokens, and `L_gpt` GPT layers.
- Space: at least `O(B * M * d_model + B * M^2 + B * M * K)` for activations/attention before optimizer state.

This is an order-of-growth estimate from the code structure, not a benchmark.

## Runtime Risks

- `from_pretrained('gpt2')` requires cached or downloadable Hugging Face GPT-2 weights.
- `CALF/wte_pca_500.pt` is loaded through `utils.torch_compat.load_numpy_torch_artifact()`, which uses a narrow safe-global allowlist for the trusted NumPy-backed artifact under modern PyTorch `weights_only=True` loading.
- Experiment checkpoints are loaded through `utils.torch_compat.load_state_dict_checkpoint()` with `map_location=self.device`.
- Several validation methods set only selected submodules to eval mode (`in_layer`, `out_layer`, `time_proj`, `text_proj`) rather than calling `self.model.eval()`. GPT-2 dropout behavior during validation should be reviewed before relying on exact metrics.
