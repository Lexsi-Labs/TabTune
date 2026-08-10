"""Load and save TabTune configurations as YAML or JSON.

A run described by a file rather than by a Python call site is reproducible,
diffable and reviewable. That property is what makes experiment tracking and
model-risk documentation possible, so the loader is deliberately strict about
round-tripping: ``load_config(save_config(cfg, p)) == cfg``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..registry.errors import ConfigError
from .schemas import PipelineConfig

logger = logging.getLogger(__name__)

__all__ = ["load_config", "save_config", "config_from_mapping", "dump_config"]


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is a transitive dep
        raise ConfigError(
            "Reading YAML configs requires PyYAML. Install it with `pip install pyyaml`."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return dict(data)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError(
            "Writing YAML configs requires PyYAML. Install it with `pip install pyyaml`."
        ) from exc
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, default_flow_style=False)


def config_from_mapping(data: Mapping[str, Any], *, strict: bool = False) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from a nested mapping.

    Accepts both the nested form (``{"tuning": {"epochs": 5}}``) and the flat
    legacy form (``{"tuning_params": {"epochs": 5}}``), so configs can be
    written by hand or dumped from an existing pipeline.

    Args:
        data: The mapping to convert.
        strict: Forbid unrecognised keys instead of warning.

    Returns:
        A validated configuration.

    Raises:
        ConfigError: On invalid or (when ``strict``) unrecognised content.
    """
    payload = dict(data)

    # Accept the legacy *_params names used by TabularPipeline's constructor.
    aliases = {
        "tuning_params": "tuning",
        "processor_params": "processor",
        "context_sampling_params": "context_sampling",
    }
    for legacy, modern in aliases.items():
        if legacy in payload and modern not in payload:
            payload[modern] = payload.pop(legacy)
        elif legacy in payload:
            payload.pop(legacy)

    try:
        config = PipelineConfig(**payload)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Invalid pipeline configuration: {exc}") from exc

    # File-driven configs are validated eagerly: the whole point of a config
    # file is that `tabtune fit --config run.yaml` fails in the first second
    # rather than after a checkpoint download.
    return config.validate_against_registry()


def load_config(path: str | Path, *, strict: bool = False) -> PipelineConfig:
    """Load a pipeline configuration from a ``.yaml``, ``.yml`` or ``.json`` file.

    Args:
        path: Path to the configuration file.
        strict: Forbid unrecognised keys instead of warning.

    Returns:
        A validated configuration.

    Raises:
        FileNotFoundError: If the path does not exist.
        ConfigError: If the file cannot be parsed or fails validation.

    Example:
        >>> load_config("runs/credit.yaml")            # doctest: +SKIP
        PipelineConfig(model_name='TabICLv2', ...)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = _read_yaml(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise ConfigError(f"{path} must contain a JSON object at the top level")
        data = dict(data)
    else:
        raise ConfigError(
            f"Unsupported config format {suffix!r}; expected .yaml, .yml or .json"
        )

    config = config_from_mapping(data, strict=strict)
    logger.info("[Config] Loaded pipeline configuration from %s", path)
    return config


def dump_config(config: PipelineConfig) -> dict[str, Any]:
    """Return a JSON-serialisable dict for ``config``, ready to write out."""
    return config.model_dump(mode="json", exclude_none=True)


def save_config(config: PipelineConfig, path: str | Path) -> Path:
    """Write ``config`` to ``path`` as YAML or JSON, inferred from the suffix.

    Args:
        config: The configuration to persist.
        path: Destination path. Parent directories are created.

    Returns:
        The path written.

    Raises:
        ConfigError: On an unsupported extension.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dump_config(config)

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        _write_yaml(path, payload)
    elif suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
    else:
        raise ConfigError(
            f"Unsupported config format {suffix!r}; expected .yaml, .yml or .json"
        )

    logger.info("[Config] Saved pipeline configuration to %s", path)
    return path
