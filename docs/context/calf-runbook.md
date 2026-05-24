# CALF Runbook

Last verified: 2026-05-23 from local repository files and compatibility checks.

## Where To Start

From the repo root, CALF examples assume:

```bash
cd CALF
sh scripts/long_term_forecasting/ETTh_GPT2.sh
sh scripts/long_term_forecasting/ETTm_GPT2.sh
sh scripts/long_term_forecasting/traffic.sh
```

The actual entrypoint is:

```bash
python run.py --task_name long_term_forecast --model GPT4TS --model_id <dataset_seq_pred_mode> --log_fine_name <log.txt> ...
```

`--log_fine_name` is required by `run.py`.

## Long-Term Forecasting Scripts

Scripts under `CALF/scripts/long_term_forecasting/` sweep:

- datasets: ETT hour/minute, weather, traffic, electricity.
- horizons: usually `96 192 336 720`.
- ablation modes: `ori`, `dropAttn_keepWE`, `llm_to_attn`, `llm_to_trsf`.

Important path expectation: these scripts use `./datasets/...` relative to `CALF/`, while the current workspace has `dataset/` singular at repo root. Fix the path or create the expected layout before running.

## Outputs And Artifacts

CALF training/testing can write:

- `checkpoints/<setting>/checkpoint.pth`
- `result_*.txt` files for task summaries
- `test_results/<setting>/`
- `results/<setting>/`
- `m4_results/<model>/`
- `bootstrap/*.npy` and `bs_results.txt` when `--bootstrap_eval` is enabled

Do not run training from the repo root unless paths are intentionally adjusted; most CALF scripts assume current directory is `CALF/`.

## Quick Source Index

| Need | File |
| --- | --- |
| CLI args and task dispatch | `CALF/run.py` |
| Base experiment/device/model registry | `CALF/exp/exp_basic.py` |
| Long-term train/test loop | `CALF/exp/exp_long_term_forecasting.py` |
| M4 short-term loop | `CALF/exp/exp_short_term_forecasting.py` |
| Imputation/classification/anomaly loops | `CALF/exp/exp_imputation.py`, `CALF/exp/exp_classification.py`, `CALF/exp/exp_anomaly_detection.py` |
| Dataset routing | `CALF/data_provider/data_factory.py` |
| ETT/custom/M4/anomaly/UEA loaders | `CALF/data_provider/data_loader.py`, `CALF/data_provider/m4.py`, `CALF/data_provider/uea.py` |
| Main model | `CALF/models/GPT4TS.py` |
| Custom GPT-2 wrapper | `CALF/models/GPT2_arch.py` |
| Ablation attention replacement | `CALF/models/Attention.py` |
| Distillation objective | `CALF/utils/distillationLoss.py` |

## Practical Checks Before Running

- Confirm a usable environment manually; `CALF/requirements.txt` is missing in this checkout.
- Confirm `wte_pca_500.pt` can be loaded under the installed PyTorch version before launching a long run.
- Confirm the dataset directory matches the script paths (`CALF/datasets/...` by default).
- Confirm GPU index flags match available devices. Long-term scripts now default to local GPU `0`; set `CUDA_VISIBLE_DEVICES` or `GPU_LOC` explicitly before running multi-GPU/server sweeps.
- Keep `model_id` mode substrings intact because CALF branches on them directly.

## Useful Search Commands

```bash
rg -n "model_id|dropAttn_keepWE|llm_to_attn|llm_to_trsf|randomInit|ori" CALF
rg -n "outputs_time|outputs_text|intermidiate" CALF
rg -n "DistillationLoss|task_w|feature_w|logits_w" CALF
rg -n "root_path|data_path|datasets|dataset" CALF README.md
```
