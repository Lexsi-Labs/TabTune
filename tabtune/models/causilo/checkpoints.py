"""Load pinned safetensors weights and verify their task/configuration pairing."""

import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .model import Model, ModelConfig

REPOSITORY = "nums-ai/causilo"
RELEASE_COMMIT = "94f2bd91db0737d4da59f347910662905ecb5a09"


def load_checkpoint(directory: Path) -> Model:
    """Validate the format and config hash, then materialize an evaluation model."""
    encoded = (directory / "config.json").read_bytes()
    record = json.loads(encoded)
    expected = {"architecture": "causilo-v1.0", "format_version": 1, "config_version": 1}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported model configuration")
    config = ModelConfig(**record["model"])
    with safe_open(directory / "model.safetensors", framework="pt") as artifact:
        metadata = artifact.metadata()
        required = {
            "architecture": "causilo-v1.0",
            "format_version": "1",
            "task": config.task,
            "config_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        if metadata != required:
            raise ValueError("Weights do not match their task and configuration")
        tensors = {key: artifact.get_tensor(key) for key in artifact.keys()}
    # Construct shapes without allocating a second full copy of the weights.
    with torch.device("meta"):
        model = Model(config)
    model.load_state_dict(tensors, strict=True, assign=True)
    return model.eval()


def load_pretrained_model(task: str) -> Model:
    """Fetch the task's files at the pinned revision, reusing the Hub cache when available."""
    folder = "classifier" if task == "classification" else "regressor"
    root = snapshot_download(
        REPOSITORY,
        revision=RELEASE_COMMIT,
        allow_patterns=[f"{folder}/config.json", f"{folder}/model.safetensors"],
    )
    model = load_checkpoint(Path(root) / folder)
    if model.config.task != task:
        raise ValueError("The official checkpoint has the wrong prediction task")
    return model
