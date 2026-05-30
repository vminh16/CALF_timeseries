# CALF BTCUSDT Workspace Overview

Last verified: 2026-05-30 from local repository files.

## Current Scope

This repository is now scoped to the CALF experiments for `BTCUSDT` forecasting.
The root `README.md` is the primary project document. It records the local GPU
setup, checkpoint evaluation pipeline, artifact validation notebook, and the
measured results.

The original workspace also contains `OFA/`, `PAttn/`, `Time-LLM-exp/`, and
`pic/`. Those directories remain available locally but are ignored and are no
longer tracked because they are outside the current research scope.

## Tracked Top-Level Map

| Path | Role |
| --- | --- |
| `CALF/` | CALF source, BTCUSDT scripts, compatibility helpers, and notebook. |
| `docs/context/` | Source audit and local environment notes. |
| `tests/` | Focused tests for the local CALF compatibility changes. |
| `README.md` | BTCUSDT experiment report and reproducible run instructions. |

## CALF Runtime Artifacts

The following directories are intentionally local-only:

| Path | Role |
| --- | --- |
| `CALF/dataset/` | Input datasets. |
| `CALF/checkpoints/` | Trained model checkpoints. |
| `CALF/logs_btc/` | Training logs. |
| `CALF/logs_btc_eval/` | Evaluation logs. |
| `CALF/results/` | Evaluation artifacts consumed by the notebook. |

## Verified CALF Flow

1. Train with `CALF/scripts/long_term_forecasting/run_btc_calf_local.bat`.
2. Evaluate `pred_len=16` checkpoints with
   `CALF/scripts/long_term_forecasting/run_btc_calf_eval_local.bat`.
3. Read the timestamped `pred_0.npy`, `true_0.npy`, `metrics_0.npy`, and
   `run_info.txt` artifacts in
   `CALF/notebooks/btc_pred16_checkpoint_evaluation.ipynb`.
4. Validate artifact alignment before plotting 96 historical steps and 16
   forecast steps.

## Reference

- Peiyuan Liu et al., *CALF: Aligning LLMs for Time Series Forecasting via
  Cross-modal Fine-Tuning*, arXiv:2403.07300:
  <https://arxiv.org/abs/2403.07300>
