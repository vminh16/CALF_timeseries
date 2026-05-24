from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CALF_ROOT = REPO_ROOT / "CALF"


def test_calf_long_term_scripts_use_parser_argument_name():
    bad_occurrences = []
    for script_path in (CALF_ROOT / "scripts" / "long_term_forecasting").glob("*.sh"):
        text = script_path.read_text(encoding="utf-8")
        if "--gpt_layer " in text:
            bad_occurrences.append(script_path.relative_to(REPO_ROOT).as_posix())

    assert bad_occurrences == []


def test_calf_word_embedding_artifact_loads_with_safe_weights_only_allowlist():
    import sys

    sys.path.insert(0, str(CALF_ROOT))
    from utils.torch_compat import load_numpy_torch_artifact

    artifact = load_numpy_torch_artifact(CALF_ROOT / "wte_pca_500.pt", map_location="cpu")

    assert artifact.shape == (768, 500)
    assert str(artifact.dtype) == "float32"


def test_calf_deprecated_torch_api_patterns_removed():
    offenders = []
    for py_path in CALF_ROOT.rglob("*.py"):
        if "__pycache__" in py_path.parts:
            continue
        text = py_path.read_text(encoding="utf-8")
        if "MSELoss(reduce=False)" in text or "functional.softmax(preds)" in text:
            offenders.append(py_path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_calf_models_do_not_import_unused_peft_dependency():
    offenders = []
    for py_path in (CALF_ROOT / "models").glob("*.py"):
        text = py_path.read_text(encoding="utf-8")
        if "from peft import" in text:
            offenders.append(py_path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_calf_transformers_imports_use_public_api_only():
    forbidden_patterns = [
        "from transformers.models",
        "BertTokenizer",
        "BertModel",
        "AutoTokenizer",
    ]
    offenders = []
    for py_path in (CALF_ROOT / "models").glob("*.py"):
        text = py_path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in text:
                offenders.append(f"{py_path.relative_to(REPO_ROOT).as_posix()}::{pattern}")

    assert offenders == []


def test_calf_imports_data_loader_without_patoolib_dependency():
    import importlib
    import sys

    sys.path.insert(0, str(CALF_ROOT))
    sys.modules.pop("data_provider.m4", None)
    sys.modules.pop("data_provider.data_loader", None)

    importlib.import_module("data_provider.data_loader")


def test_calf_python312_numpy2_removed_patterns_are_absent():
    forbidden_patterns = [
        "from distutils",
        "import distutils",
        "np.Inf",
        "np.Infinity",
        "np.NaN",
        "np.NINF",
    ]
    offenders = []
    for py_path in CALF_ROOT.rglob("*.py"):
        if "__pycache__" in py_path.parts:
            continue
        text = py_path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in text:
                offenders.append(f"{py_path.relative_to(REPO_ROOT).as_posix()}::{pattern}")

    assert offenders == []


def test_calf_experiments_use_safe_checkpoint_loader():
    offenders = []
    for py_path in (CALF_ROOT / "exp").glob("*.py"):
        text = py_path.read_text(encoding="utf-8")
        if "torch.load(" in text:
            offenders.append(py_path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_calf_no_hardcoded_multi_gpu_visibility_or_device_allocations():
    offenders = []
    for path in CALF_ROOT.rglob("*"):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8")
        if 'CUDA_VISIBLE_DEVICES"] = "0,1,2"' in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
        if 'CUDA_VISIBLE_DEVICES="0,1,2"' in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
        if "gpu_loc=1" in text or "gpu_loc=2" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
        if ".cuda()" in text:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_calf_gpt2_wrapper_delegates_to_transformers_forward():
    text = (CALF_ROOT / "models" / "GPT2_arch.py").read_text(encoding="utf-8")

    assert "super().forward" in text
    assert "layer_past=" not in text
    assert "warn_if_padding_and_no_attention_mask" not in text
    assert "torch.utils.checkpoint.checkpoint" not in text


def test_calf_avoids_tensor_data_escape_hatches():
    forbidden_patterns = [
        "forecast.data",
        "target.data",
        "L2_distance.data",
        "param.data",
        "target_wpe_param_.data",
    ]
    offenders = []
    for py_path in CALF_ROOT.rglob("*.py"):
        if "__pycache__" in py_path.parts:
            continue
        text = py_path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in text:
                offenders.append(f"{py_path.relative_to(REPO_ROOT).as_posix()}::{pattern}")

    assert offenders == []
