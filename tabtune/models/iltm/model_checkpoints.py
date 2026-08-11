import os
import sys
from pathlib import Path
from typing import Dict, Any

from huggingface_hub import hf_hub_download


CKPT_DIR_ENV = "ILTM_CKPT_DIR"
HF_REPO_ID = "dbonet/iLTM"


def _get_platform_cache_dir() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "iltm"
        return Path.home() / ".iltm"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "iltm"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            return Path(xdg_cache) / "iltm"
        return Path.home() / ".cache" / "iltm"


def _get_default_ckpt_dir() -> str:
    if CKPT_DIR_ENV in os.environ:
        ckpt_dir = os.environ[CKPT_DIR_ENV]
    else:
        ckpt_dir = str(_get_platform_cache_dir())
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def _ensure_checkpoint(repo_id: str, filename: str, ckpt_dir: str) -> str:
    local_path = os.path.join(ckpt_dir, filename)
    if os.path.isfile(local_path):
        return local_path
    
    try:
        from huggingface_hub.utils import HfHubHTTPError
        
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=ckpt_dir,
        )
        return path
    except HfHubHTTPError as e:
        if e.response.status_code in (401, 403):
            error_msg = (
                f"\nFailed to download checkpoint '{filename}' from Hugging Face.\n"
                f"Repository: {repo_id}\n\n"
                f"This appears to be an authentication issue. Try the following:\n\n"
                f"  1. Authenticate with Hugging Face using one of these commands:\n"
                f"     - huggingface-cli login\n"
                f"     - hf auth login\n\n"
                f"  2. If you don't have an account, create one at:\n"
                f"     https://huggingface.co/join\n\n"
            )
            raise RuntimeError(error_msg) from e
        raise
    except ImportError:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=ckpt_dir,
        )
        return path


def get_model_checkpoint_config(model_name_suffix: str) -> Dict[str, Any]:
    ckpt_dir = _get_default_ckpt_dir()

    registry: Dict[str, Dict[str, Any]] = {}
    registry_configs = {
        "r128bn": {
            "preprocessing": "realmlp_td_s_v0",
            "bottleneck_size": 128,
            "tree_embedding": False,
        },
        "rnobn": {
            "preprocessing": "realmlp_td_s_v0",
            "tree_embedding": False,
        },
        "catb": {
            "preprocessing": "none",
            "tree_embedding": True,
            "tree_model": "CatBoost",
        },
        "xgb": {
            "preprocessing": "none",
            "tree_embedding": True,
            "tree_model": "XGBoost_hist",
        },
        "rtr": {
            "preprocessing": "realmlp_td_s_v0",
            "do_retrieval": True,
            "tree_embedding": False,
        },
        "rtrcb": {
            "preprocessing": "none",
            "tree_embedding": True,
            "tree_model": "CatBoost",
            "do_retrieval": True,
        },
        "cbrconcat": {
            "preprocessing": "realmlp_td_s_v0",
            "tree_embedding": True,
            "tree_model": "CatBoost",
            "concat_tree_with_orig_features": True,
        },
        "xgbrconcat": {
            "preprocessing": "realmlp_td_s_v0",
            "tree_embedding": True,
            "tree_model": "XGBoost_hist",
            "concat_tree_with_orig_features": True,
        },
    }
    for model_name, cfg in registry_configs.items():
        registry[model_name] = {
            "filename": f"{model_name}.pth",
            **cfg,
        }

    for model_name, cfg in registry.items():
        if model_name_suffix.endswith(model_name):
            filename = cfg["filename"]
            local_path = _ensure_checkpoint(HF_REPO_ID, filename, ckpt_dir)
            out = {k: v for k, v in cfg.items() if k != "filename"}
            out["checkpoint"] = local_path
            return out

    available = list(registry.keys())
    raise ValueError(
        f"Unknown model name suffix: {model_name_suffix}. Available models: {available}"
    )


def resolve_model_checkpoint(checkpoint: str) -> Dict[str, Any]:
    if os.path.isfile(checkpoint):
        return {"checkpoint": checkpoint}

    if os.path.isdir(checkpoint):
        pth_files = list(Path(checkpoint).glob("*.pth"))
        if pth_files:
            return {"checkpoint": str(pth_files[0])}

    return get_model_checkpoint_config(checkpoint)

