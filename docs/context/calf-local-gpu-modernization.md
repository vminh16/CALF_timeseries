# CALF Local GPU Modernization Notes

Last verified: 2026-05-23.

## Local Verdict

The current Python environment cannot run CALF on the local GPU:

- `nvidia-smi` sees `NVIDIA GeForce RTX 2050` with `4096 MiB` VRAM and driver CUDA capability `12.7`.
- Python is `3.12.11`.
- Installed PyTorch is `torch 2.9.1+cpu`; `torch.cuda.is_available()` returns `False`.
- Required CALF model imports are still missing in the current base environment: `transformers` and `einops`. `patoolib` is no longer required by CALF because the import in the M4 module was unused. `peft` imports were removed from the active CALF model path because the LoRA wrapper was not applied.

This means local GPU deployment is not possible in the current environment. It is feasible to make small local GPU smoke runs after installing a CUDA-enabled PyTorch wheel and CALF dependencies, but the original sweep scripts are not realistic as-is on 4GB VRAM. The ETT/weather cases may be tested with reduced batch size and fewer GPT layers; traffic/electricity are high-risk because CALF treats variables as tokens and traffic has hundreds of channels.

## Authoritative References Checked

- PyTorch install docs: Windows supports Python 3.10-3.14, and CUDA-enabled pip installs should use the official PyTorch selector.
  https://pytorch.org/get-started/locally/
- PyTorch previous-version matrix: official CUDA wheel commands exist for Windows/Linux, including CUDA 12.6 and CUDA 12.8 builds for recent PyTorch versions.
  https://pytorch.org/get-started/previous-versions/
- PyTorch serialization docs: from PyTorch 2.6 onward, `torch.load` defaults to `weights_only=True` unless a custom pickle module is passed.
  https://docs.pytorch.org/docs/main/notes/serialization.html#torch-load-with-weights-only-true
- Hugging Face GPT-2 docs/model card: `GPT2Model` supports `inputs_embeds`, and the `gpt2` checkpoint is the 124M-parameter small GPT-2.
  https://huggingface.co/docs/transformers/main/model_doc/gpt2
  https://huggingface.co/openai-community/gpt2
- PEFT LoRA docs: `LoraConfig` is only useful when used to create/adapt a LoRA model; CALF had created the config but commented out `get_peft_model`.
  https://huggingface.co/docs/peft/main/en/package_reference/lora
- PyTorch MSELoss docs: `reduce` is deprecated in favor of `reduction`.
  https://docs.pytorch.org/docs/2.12/generated/torch.nn.modules.loss.MSELoss.html
- Python 3.12 docs: `distutils` was removed from the standard library after deprecation in Python 3.10.
  https://docs.python.org/3.12/library/distutils.html
- NumPy 2.0 migration guide: removed namespace aliases such as `np.Inf` should be replaced with `np.inf`.
  https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html

## Methods Considered

### Method A: Preserve old environment

Install versions close to the original project lineage, likely Python 3.9 and older PyTorch/Transformers. Complexity is low for code changes, but high for environment fragility. It also preserves old API usage such as unsafe/default `torch.load` assumptions and deprecated loss options.

Time complexity of code work: `O(1)` edits. Practical viability: weak on a modern Windows/Python 3.12 setup.

### Method B: Modernize CALF for current PyTorch/Transformers

Keep the architecture intact and patch compatibility points: safe `torch.load` for NumPy-backed artifacts, script arg names, deprecated PyTorch APIs, and unused hard dependencies. Complexity is `O(N)` over source files scanned and modified. Practical viability: best for maintaining this repo on current local machines.

Chosen path: Method B.

## Changes Applied

- Added `CALF/utils/torch_compat.py` with `load_numpy_torch_artifact()` using `torch.serialization.safe_globals` so `CALF/wte_pca_500.pt` loads under PyTorch 2.6+ without using `weights_only=False`.
- Updated `CALF/models/GPT4TS.py` and `CALF/models/GPT4TS_ETT.py` to use that loader.
- Removed unused `peft` imports and the unused `LoraConfig` block from CALF model files.
- Removed the unused `patoolib` import from the M4 data module so importing `data_provider.data_loader` no longer needs that package.
- Replaced `distutils.util.strtobool` with a local `str_to_bool()` helper for Python 3.12 compatibility.
- Replaced NumPy 2-incompatible `np.Inf` with `np.inf`.
- Changed `nn.MSELoss(reduce=False)` to `nn.MSELoss(reduction='none')`.
- Changed classification softmax calls to pass `dim=1`.
- Fixed all CALF long-term scripts from `--gpt_layer` to `--gpt_layers`.
- Added a safe state-dict checkpoint loader and routed experiment checkpoint loads through `torch.load(..., weights_only=True, map_location=self.device)`.
- Replaced hard-coded long-term script GPU visibility/index defaults with local-friendly defaults: `CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"` and `GPU_LOC` override support where applicable.
- Replaced device-specific `.cuda()` tensor creation in distillation utilities with tensors allocated on the source/target device.
- Replaced tensor `.data` mutation/reads with `copy_()` under `torch.no_grad()` or `.detach()` where the code only needs non-gradient values.
- Replaced the copied old Transformers GPT-2 forward implementation with a thin wrapper over `super().forward(...)` that preserves CALF's `(last_hidden_state, hidden_states)` return contract.
- Switched GPT-2 imports to the public `transformers` API and removed unused tokenizer/BERT imports.
- Added `CALF/requirements-modern.txt` for non-PyTorch CALF dependencies. Install CUDA PyTorch separately using the official PyTorch selector.

## Suggested Local GPU Smoke Plan

1. Create a fresh environment rather than changing the base Conda env.
2. Install a CUDA-enabled PyTorch wheel from the official PyTorch command for Windows/Pip/CUDA 12.6 or CUDA 12.8.
3. Install `CALF/requirements-modern.txt`.
4. Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest tests/test_calf_modernization.py -q
```

5. Run a minimal ETT smoke only, with reduced settings:

```bash
cd CALF
python run.py --root_path ../dataset/ETT-small/ --data_path ETTh1.csv --is_training 1 --task_name long_term_forecast --model_id ETTh1_96_96_dropAttn_keepWE --data ETTh1 --seq_len 96 --label_len 0 --pred_len 96 --batch_size 1 --learning_rate 0.0005 --train_epochs 1 --d_model 768 --n_heads 4 --d_ff 768 --dropout 0.3 --enc_in 7 --c_out 7 --gpt_layers 1 --itr 1 --model GPT4TS --gpu 0 --patience 1 --log_fine_name local_smoke.txt
```

This smoke command intentionally uses `dropAttn_keepWE` and `batch_size 1` to reduce GPU memory pressure. The original full sweeps should be treated as cloud/server GPU workloads.
