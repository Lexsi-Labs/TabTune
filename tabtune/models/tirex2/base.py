# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Loading utilities for inference-ready :class:`TiRex2` checkpoints."""

from pathlib import Path
from typing import Any

import torch
import yaml

from .api_adapter import ForecastModel
from .model import TiRex2
from .model.component.flashrnn_slstm import _FlashRNNLayer
from .model.component.mlstm_block import mLSTMLayer
from .model.component.residual_block import ResidualBlock

CONFIG_FILENAME = "model-config.yaml"
CKPT_FILENAME = "model.ckpt"


def _resolve_ckpt_dir(
    ckpt_path: str | Path,
    hf_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Resolve a local checkpoint directory or download one from Hugging Face."""
    raw_path = str(ckpt_path)
    local_path = Path(raw_path).expanduser()
    if local_path.is_dir():
        return local_path

    if raw_path.startswith("hf://"):
        repo_id = raw_path.removeprefix("hf://")
    elif not local_path.exists() and _looks_like_hf_repo_id(raw_path):
        repo_id = raw_path
    else:
        return local_path

    from huggingface_hub import snapshot_download

    hf_kwargs = hf_kwargs or {}
    return Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[CONFIG_FILENAME, CKPT_FILENAME],
            **hf_kwargs,
        )
    )


def _looks_like_hf_repo_id(path: str) -> bool:
    """Heuristic for Hugging Face repo ids like ``org/model-name``."""
    return not path.startswith((".", "/", "~")) and path.count("/") == 1


def load_model(
    ckpt_path: str | Path = "NX-AI/TiRex-2",
    device: str = "cpu",
    *,
    hf_kwargs: dict[str, Any] | None = None,
    compile: bool = False,
) -> ForecastModel:
    """Load an inference-ready :class:`TiRex2` from a checkpoint directory or HF repo.

    Parameters
    ----------
    ckpt_path : str or pathlib.Path
        Local directory holding ``model-config.yaml`` and ``model.ckpt``. Values
        of the form ``hf://org/repo`` or ``org/repo`` are treated as Hugging Face
        model repo ids and downloaded with :func:`huggingface_hub.snapshot_download`.
    device : {"cpu", "cuda", "mps"}
        Runtime device (``"cuda:N"`` is accepted). This overrides any
        device/backend stored in the checkpoint config. The recurrent kernels
        are pure PyTorch and identical on every device.
    hf_kwargs : dict, optional
        Extra keyword arguments forwarded to ``snapshot_download`` for Hugging
        Face paths, e.g. ``{"revision": "main", "local_files_only": True}``.
    compile : bool
        If True, ``torch.compile`` the mLSTM and sLSTM layers and the residual
        input/output blocks.

    Returns
    -------
    ForecastModel
        The instantiated backbone (with the checkpoint weights loaded, set to
        evaluation mode) wrapped in a :class:`ForecastModel` that exposes the
        high-level ``forecast`` API.

    Examples
    --------
    >>> import torch
    >>> from tabtune.models.tirex2 import TimeseriesType, load_model
    >>> model = load_model("NX-AI/TiRex-2", device="cpu")
    >>> ts = TimeseriesType(target=torch.randn(1, 128), past_covariates=None, future_covariates=None)
    >>> forecast = model.forecast([ts], prediction_length=32, output_type="numpy")[0]
    >>> forecast.shape
    (1, 9, 32)
    """
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Execution on CUDA was requested but is not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Execution on MPS was requested but is not available.")

    ckpt_dir = _resolve_ckpt_dir(ckpt_path, hf_kwargs=hf_kwargs)
    config_file = ckpt_dir / CONFIG_FILENAME
    weights_file = ckpt_dir / CKPT_FILENAME
    if not config_file.is_file():
        raise FileNotFoundError(f"Expected model config at {config_file}")
    if not weights_file.is_file():
        raise FileNotFoundError(f"Expected model checkpoint at {weights_file}")

    with config_file.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)

    config["device"] = device
    model = TiRex2(**config)

    checkpoint = torch.load(weights_file, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if compile:
        for module in model.modules():
            if isinstance(module, (mLSTMLayer, _FlashRNNLayer, ResidualBlock)):
                module.compile()

    return ForecastModel(model)
