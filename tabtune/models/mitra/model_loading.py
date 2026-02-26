"""Functions for downloading and loading Mitra model checkpoints from HuggingFace."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from urllib.error import URLError

try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_hub_download = None

logger = logging.getLogger(__name__)

# HuggingFace repository for Mitra regressor
MITRA_REGRESSOR_REPO = "autogluon/mitra-regressor"
MITRA_REGRESSOR_FILES = ["model.safetensors", "config.json"]


def _try_hf_hub_download(
    base_path: Path,
    repo_id: str,
    filename: str,
) -> None:
    """Try to download model files using HuggingFace Hub."""
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install it with: pip install huggingface_hub"
        )

    logger.info(f"[MitraModelLoader] Attempting HuggingFace download: {filename}")

    try:
        # Download to a temporary location first
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=base_path.parent,
        )
        # Move file to desired location
        if Path(local_path) != base_path:
            if base_path.exists():
                base_path.unlink()
            Path(local_path).rename(base_path)

        logger.info(f"[MitraModelLoader] Successfully downloaded {filename} to {base_path}")
    except Exception as e:
        raise Exception(f"HuggingFace download failed for {filename}!") from e


def _try_direct_download(
    base_path: Path,
    repo_id: str,
    filename: str,
) -> None:
    """Try to download model files using direct URLs."""
    model_url = (
        f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true"
    )

    # Create parent directory if it doesn't exist
    base_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[MitraModelLoader] Attempting direct download from {model_url}")

    try:
        with urllib.request.urlopen(model_url) as response:  # noqa: S310
            if response.status != 200:
                raise URLError(
                    f"HTTP {response.status} when downloading from {model_url}",
                )
            base_path.write_bytes(response.read())

        logger.info(f"[MitraModelLoader] Successfully downloaded {filename} to {base_path}")
    except Exception as e:
        raise Exception(f"Direct download failed for {filename}!") from e


def download_mitra_regressor(
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download Mitra regressor model from HuggingFace.

    Args:
        cache_dir: Directory to cache the model. If None, uses default cache location.
        force_download: If True, force re-download even if model exists.

    Returns:
        Path to the directory containing the downloaded model files.

    Raises:
        Exception: If download fails from all sources.
    """
    if cache_dir is None:
        # Use default cache location similar to TabPFN
        cache_dir = Path.home() / ".cache" / "mitra"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir = cache_dir / "mitra-regressor"

    # Check if model already exists
    if not force_download and model_dir.exists():
        model_file = model_dir / "model.safetensors"
        config_file = model_dir / "config.json"
        if model_file.exists() and config_file.exists():
            logger.info(f"[MitraModelLoader] Model already exists at {model_dir}")
            return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)

    # Download model files
    errors = []
    for filename in MITRA_REGRESSOR_FILES:
        file_path = model_dir / filename

        # Try HuggingFace Hub first
        if HF_HUB_AVAILABLE:
            try:
                _try_hf_hub_download(file_path, MITRA_REGRESSOR_REPO, filename)
                continue
            except Exception as e:
                errors.append(f"HuggingFace Hub download failed: {e}")
                logger.warning(f"[MitraModelLoader] HuggingFace Hub download failed: {e}")

        # Fallback to direct download
        try:
            _try_direct_download(file_path, MITRA_REGRESSOR_REPO, filename)
        except Exception as e:
            errors.append(f"Direct download failed: {e}")
            logger.error(f"[MitraModelLoader] Direct download failed: {e}")
            raise Exception(
                f"Failed to download {filename} from all sources!\n"
                f"Errors: {errors}\n\n"
                f"Please download manually from:\n"
                f"https://huggingface.co/{MITRA_REGRESSOR_REPO}/resolve/main/{filename}\n"
                f"Then place it at: {file_path}"
            ) from e

    logger.info(f"[MitraModelLoader] Successfully downloaded Mitra regressor to {model_dir}")
    return model_dir


def load_mitra_regressor_from_hf(
    cache_dir: str | Path | None = None,
    device: str = "cuda",
    force_download: bool = False,
):
    """Download and load Mitra regressor model from HuggingFace.

    Args:
        cache_dir: Directory to cache the model. If None, uses default cache location.
        device: Device to load the model on ('cuda' or 'cpu').
        force_download: If True, force re-download even if model exists.

    Returns:
        Loaded Tab2D model instance configured for regression.
    """
    from tabtune.models.mitra.tab2d import Tab2D

    # Download model
    model_dir = download_mitra_regressor(cache_dir=cache_dir, force_download=force_download)

    # Load using from_pretrained
    model = Tab2D.from_pretrained(str(model_dir), device=device)

    logger.info(f"[MitraModelLoader] Successfully loaded Mitra regressor from {model_dir}")
    return model
