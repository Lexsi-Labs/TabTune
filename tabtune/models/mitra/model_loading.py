"""Functions for downloading and loading Mitra model checkpoints from HuggingFace."""

from __future__ import annotations

import logging
import os
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

# HuggingFace repository for the Mitra *classifier*. This mirrors the regressor
# constant above and matches ``ModelSpec(name="Mitra").weights`` in the registry.
# Before this existed, nothing in the codebase ever consumed that declared
# weights string, so the classification path silently ran on randomly
# initialised weights.
MITRA_CLASSIFIER_REPO = "autogluon/mitra-classifier"
MITRA_CLASSIFIER_FILES = ["model.safetensors", "config.json"]

# ---------------------------------------------------------------------------
# Mitra v2
#
# v2 is a CHECKPOINT, not a new architecture, and the same vendored ``Tab2D``
# loads it. That is a property of the format rather than a guess:
# ``Tab2D.save_pretrained`` writes exactly five keys -- dim, dim_output,
# n_layers, n_heads, task -- ``from_pretrained`` reads exactly those five and
# builds the module from them, and ``Tab2D.__init__`` takes no other
# architectural argument. So there is no shape-neutral difference a v2
# checkpoint could carry that would load silently wrong, and anything
# structural is caught by the strict ``load_state_dict`` below.
#
# Because the class is shared, a v2 model IS a ``Tab2D``, so every
# ``isinstance(model, Tab2D)`` branch in the pipeline and the TuningManager --
# fine-tuning, PEFT, predict, predict_proba -- covers it with no second
# dispatch arm. Vendoring a second copy of the tree would duplicate 1,100
# lines AND break those branches, since a second class object is not the same
# class.
#
# The repo ids are overridable from the environment: they could not be
# verified from the machine this was written on (huggingface.co is blocked by
# its egress policy), so a rename must not require editing vendored source.
MITRA_V2_CLASSIFIER_REPO = os.environ.get(
    "TABTUNE_MITRA_V2_CLS_REPO", "autogluon/mitra-classifier-2"
)
MITRA_V2_REGRESSOR_REPO = os.environ.get(
    "TABTUNE_MITRA_V2_REG_REPO", "autogluon/mitra-regressor-2"
)

#: A separate checkpoint AutoGluon publishes alongside the two above. What it
#: is meant for is NOT documented in any source reachable from here -- the name
#: suggests a fine-tuning starting point rather than a zero-shot predictor, but
#: that is an inference from the name, not a verified fact. It is exposed so it
#: can be selected, and deliberately not made anyone's default.
MITRA_FINETUNE_REPO = os.environ.get(
    "TABTUNE_MITRA_FINETUNE_REPO", "autogluon/mitra-finetune"
)

#: variant -> {task -> repo id}. ``resolve_mitra_repo`` is the single place
#: that maps a TabTune model name onto a checkpoint.
MITRA_REPOS: dict[str, dict[str, str]] = {
    "v1": {
        "classification": MITRA_CLASSIFIER_REPO,
        "regression": MITRA_REGRESSOR_REPO,
    },
    "v2": {
        "classification": MITRA_V2_CLASSIFIER_REPO,
        "regression": MITRA_V2_REGRESSOR_REPO,
    },
    "finetune": {
        "classification": MITRA_FINETUNE_REPO,
        "regression": MITRA_FINETUNE_REPO,
    },
}

#: TabTune model name -> variant key.
MITRA_MODEL_VARIANTS: dict[str, str] = {"Mitra": "v1", "MitraV2": "v2"}


def resolve_mitra_repo(model_name: str, task_type: str, variant: str | None = None) -> str:
    """The HuggingFace repo id for a Mitra model name and task.

    Args:
        model_name: ``"Mitra"`` or ``"MitraV2"``.
        task_type: ``"classification"`` or ``"regression"``.
        variant: Overrides the variant the name implies. ``"finetune"`` selects
            the separate fine-tuning checkpoint for either task.

    Returns:
        The repo id, which callers pass to ``Tab2D.from_pretrained``.

    Raises:
        ValueError: If the variant or task is unknown. Both are spelled out
            rather than defaulted, because silently falling back to v1 is how a
            v2 run would report v1 numbers.
    """
    key = variant or MITRA_MODEL_VARIANTS.get(model_name)
    if key is None:
        raise ValueError(
            f"Unknown Mitra model name {model_name!r}. "
            f"Known: {sorted(MITRA_MODEL_VARIANTS)}."
        )
    if key not in MITRA_REPOS:
        raise ValueError(
            f"Unknown Mitra variant {key!r}. Known: {sorted(MITRA_REPOS)}."
        )
    task = "regression" if str(task_type).lower().startswith("reg") else "classification"
    return MITRA_REPOS[key][task]


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


def load_mitra_classifier_from_hf(
    device: str = "cuda",
    repo_id: str = MITRA_CLASSIFIER_REPO,
):
    """Download and load the pretrained Mitra **classifier** from HuggingFace.

    The counterpart of :func:`load_mitra_regressor_from_hf`. ``Tab2D.from_pretrained``
    already resolves a HuggingFace repo id, downloads ``config.json`` +
    ``model.safetensors``, builds the module from the *checkpoint's own* config and
    then loads the state dict -- so the architecture always matches the weights.

    Note the returned model's ``dim_output`` comes from the checkpoint, not from the
    number of classes in your dataset. Mitra is trained with a fixed-width
    classification head and the head is sliced to the task's class count at
    prediction time; rebuilding the head to fit the dataset would throw the
    pretrained weights away, which is the bug this function exists to prevent.

    Args:
        device: Device to load the model on ('cuda' or 'cpu').
        repo_id: HuggingFace repo id. Defaults to ``autogluon/mitra-classifier``.

    Returns:
        A :class:`~tabtune.models.mitra.tab2d.Tab2D` configured for classification
        with pretrained weights loaded.
    """
    from tabtune.models.mitra.tab2d import Tab2D

    logger.info(f"[MitraModelLoader] Loading pretrained Mitra classifier from {repo_id}")
    model = Tab2D.from_pretrained(repo_id, device=device)
    logger.info(
        "[MitraModelLoader] Loaded Mitra classifier (dim_output=%s) from %s",
        getattr(model, "dim_output", "?"), repo_id,
    )
    return model

def probe_mitra_checkpoint(repo_or_path: str, device: str = "cpu") -> dict:
    """Report whether a Mitra checkpoint loads into the vendored ``Tab2D``.

    Answers the only question a new Mitra release raises: is it a checkpoint the
    existing code can load, or does it need new architecture code? The answer is
    decidable because ``Tab2D``'s format is closed - ``save_pretrained`` writes
    exactly ``dim``, ``dim_output``, ``n_layers``, ``n_heads`` and ``task``,
    ``from_pretrained`` reads exactly those five, and ``load_state_dict`` is
    strict - so a checkpoint either fits or raises.

    Args:
        repo_or_path: A HuggingFace repo id or a local directory holding
            ``config.json`` + ``model.safetensors``.
        device: Where to materialise the model for the load test.

    Returns:
        A dict with ``verdict``:

        - ``"checkpoint_swap"``  the vendored Tab2D loaded it; nothing to do.
        - ``"needs_new_code"``   it was fetched and did not fit the architecture.
        - ``"unreachable"``      it could not be fetched at all, which says
          nothing either way about the architecture and must not be read as if
          it did.

        plus the checkpoint's ``config``, any ``unexpected_config_keys`` beyond
        the five the loader consumes, and ``error`` when something failed.
    """
    import json

    from tabtune.models.mitra.tab2d import Tab2D

    known = {"dim", "dim_output", "n_layers", "n_heads", "task"}
    result: dict = {
        "source": repo_or_path,
        "verdict": "unreachable",
        "config": None,
        "unexpected_config_keys": [],
        "error": None,
    }

    try:
        if Path(repo_or_path).is_dir():
            config_path = Path(repo_or_path) / "config.json"
        else:
            if not HF_HUB_AVAILABLE:
                raise ImportError("huggingface_hub is required to probe a repo id.")
            config_path = Path(hf_hub_download(repo_id=repo_or_path, filename="config.json"))
        config = json.loads(Path(config_path).read_text())
        result["config"] = config
        # Extra keys are reported rather than treated as failure: the loader
        # ignores them, so they are a signal to read the model card, not proof
        # that anything is wrong.
        result["unexpected_config_keys"] = sorted(set(config) - known)
        missing = sorted(known - set(config))
        if missing:
            raise KeyError(f"config.json is missing {missing}, which from_pretrained requires")

        # Everything from here is a statement about the architecture rather
        # than about the network: the files are already local.
        result["verdict"] = "needs_new_code"
        model = Tab2D.from_pretrained(repo_or_path, device=device)
        result["verdict"] = "checkpoint_swap"
        result["loaded_shape"] = {
            "dim": model.dim, "dim_output": model.dim_output,
            "n_layers": model.n_layers, "n_heads": model.n_heads,
            "task": str(model.task),
        }
        result["n_parameters"] = sum(p.numel() for p in model.parameters())
    except Exception as exc:  # noqa: BLE001 - the exception text IS the finding
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result
