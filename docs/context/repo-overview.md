# Repo Overview

Last verified: 2026-05-22 from local repository files only.

## Project Purpose

This repository is the codebase for "Are Language Models Actually Useful for Time Series Forecasting?" It compares and ablates LLM-based time-series forecasting methods and includes `PAttn` as a simple attention baseline. The top-level README says the ablation targets are `CALF`, `OFA`/One-Fits-All, and `Time-LLM`, with `PAttn` used as the non-LLM baseline.

## Top-Level Map

| Path | Role |
| --- | --- |
| `CALF/` | CALF/LLaTA-style cross-modal knowledge distillation code. This is the main focus for this context. |
| `OFA/` | One-Fits-All/GPT4TS adaptation and ablation scripts. |
| `PAttn/` | Simple patched attention baseline introduced by this repo. |
| `Time-LLM-exp/` | Time-LLM experiment code with its own `requirements.txt` and training scripts. |
| `dataset/` | Local data checkout in this workspace; currently untracked by git. |
| `pic/` | README image assets. |

## Verified Sources Used

- `README.md`: project framing, ablation targets, PAttn motivation, run examples.
- `CALF/README.md`: CALF paper framing, expected datasets, `wte_pca_500.pt`, training artifact locations.
- `CALF/run.py`: task dispatch, CLI args, checkpoint/log behavior.
- `CALF/models/GPT4TS.py` and `CALF/models/GPT2_arch.py`: CALF model internals.
- `CALF/exp/*.py`: train/validation/test loops by task.
- `CALF/data_provider/*.py`: dataset routing and loader behavior.
- `CALF/scripts/long_term_forecasting/*.sh`: long-term CALF ablation commands.

## Analysis Approach

Two possible approaches were considered:

- Static source audit: read README, scripts, entrypoints, model files, experiment loops, and data providers. Time complexity is `O(F + L)` for files and lines scanned; space is `O(K)` for retained notes. This is the approach used because it avoids GPU, dependency, dataset, and checkpoint side effects.
- Runtime audit: import modules or run training/test scripts. Practical cost is dominated by model download, GPU memory, dataset size, epochs, and checkpoint writes; space can include checkpoints/results. This was not used because the task asked for reusable context docs, not reproduction, and CALF environment dependencies are not fully present in this checkout.

## Current Workspace Notes

- Git status before writing docs showed untracked `dataset/` and `__MACOSX/`.
- CALF scripts and README refer to `./datasets/...`, but this checkout has `dataset/` singular at repo root. Running scripts from `CALF/` will require matching the expected `CALF/datasets/...` paths or editing script paths.
- `CALF/README.md` says to install `requirements.txt`, but there is no `CALF/requirements.txt` in this checkout. Only `Time-LLM-exp/requirements.txt` exists.
- Many `__pycache__` files are tracked in git, so they should not be deleted unless cleanup is explicitly requested.
