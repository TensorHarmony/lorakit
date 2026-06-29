from pathlib import Path

import pytest


def _make_minimal_config():
    """Return a minimal config dict that passes TrainJob.__init__ validation."""
    return {
        "device": "cuda:0",
        "allow_tf32": False,
        "train": {
            "dtype": "fp32",
            "batch_size": 1,
            "resolution": 64,
            "gradient_accumulation_steps": 1,
            "dataset_folder": "/fake",
            "optimizer": "adamw",
            "optimizer_params": {"lr": 1e-4},
            "lr_scheduler": "constant",
            "lr_scheduler_params": {"num_warmup_steps": 0, "num_training_steps": 1},
            "lora": {
                "target_modules": {"to_k": [4, 4], "to_q": [4, 4]},
            },
        },
        "model": {"name_or_path": "stabilityai/stable-diffusion-xl-base-1.0"},
        "class_prompt": "man",
        "instant_prompt": "SKS",
    }


def test_arch_check_raises_on_unsupported_gpu(monkeypatch):
    """The arch check should raise RuntimeError when GPU CC is not in PyTorch's arch list."""

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda idx=0: (12, 0))
    monkeypatch.setattr(
        "torch.cuda.get_arch_list",
        lambda: ["sm_75", "sm_80", "sm_86", "sm_90"],
    )
    monkeypatch.setattr(
        "torch.cuda.get_device_name", lambda idx=0: "NVIDIA GeForce RTX 5060 Ti"
    )

    # Prevent TrainJob.__init__ from checking that the dataset folder exists.
    from lorakit import config as _config

    monkeypatch.setattr(
        _config, "resolve_user_path", lambda path, **kw: Path(path)
    )

    from lorakit.train import TrainJob

    with pytest.raises(RuntimeError, match="is not supported by this PyTorch build"):
        TrainJob(_make_minimal_config(), "1.0", "test_arch", "/tmp")


def test_arch_check_passes_on_supported_gpu(monkeypatch):
    """The arch check should not raise when GPU CC is in PyTorch's arch list."""

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda idx=0: (9, 0))
    monkeypatch.setattr(
        "torch.cuda.get_arch_list",
        lambda: ["sm_75", "sm_80", "sm_86", "sm_90"],
    )
    monkeypatch.setattr("torch.cuda.get_device_name", lambda idx=0: "NVIDIA H100")

    from lorakit import config as _config

    monkeypatch.setattr(
        _config, "resolve_user_path", lambda path, **kw: Path(path)
    )

    from lorakit.train import TrainJob

    # Should not raise
    TrainJob(_make_minimal_config(), "1.0", "test_arch", "/tmp")
