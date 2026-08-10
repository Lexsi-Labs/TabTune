"""Resolve released or user-supplied checkpoint weights to a local file path.

Resolution precedence, from most to least specific:

1. an explicit ``weights`` argument (a local file path or a Hub repo id),
2. the checkpoint's environment override (e.g. ``EXAONETABULAR_CLASSIFIER_WEIGHTS``),
3. the baked-in released coordinates, fetched from the Hub.

Only case (3) returns ``is_default=True``; that is the sole path for which the
released SHA-256 pin is enforced. When the caller supplies their own weights there
is no canonical digest to check against, so the pin is deliberately not applied —
the estimator announces that at load time.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .presets import ReleasedCheckpoint

__all__ = ["WeightSource", "resolve_weights"]

# owner/name — a single slash, Hub-legal characters, no path separators.
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class WeightSource:
    """A resolved local weights path plus whether it is the released default."""

    path: str
    is_default: bool


def _hf_download(repo_id: str, filename: str, revision: str | None, cache_dir):
    """Fetch one file from the Hub.

    The import is local so that importing this package — and resolving weights that
    live on the local filesystem — does not pull in the Hub client and its network
    stack, matching the lazy-import philosophy used elsewhere in the package.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir
    )


def _resolve_user_source(
    source: str, filename: str, revision, cache_dir
) -> WeightSource:
    if not isinstance(source, str) or not source:
        raise TypeError("weights must be a non-empty string path or Hub repo id")
    expanded = os.path.expanduser(source)
    if os.path.exists(expanded):
        if not os.path.isfile(expanded):
            raise IsADirectoryError(
                f"weights path is not a file: {expanded}"
            )
        return WeightSource(path=expanded, is_default=False)

    repo = source[len("hf://") :] if source.startswith("hf://") else source
    if not _REPO_ID.match(repo):
        raise ValueError(
            f"weights must be an existing local file or a Hub repo id "
            f"(owner/name); got {source!r}. For a custom filename in a repo, "
            f"pass filename= explicitly."
        )
    return WeightSource(
        path=_hf_download(repo, filename, revision, cache_dir), is_default=False
    )


def resolve_weights(
    checkpoint: ReleasedCheckpoint,
    weights: str | os.PathLike | None = None,
    *,
    filename: str | None = None,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
) -> WeightSource:
    """Resolve ``weights`` to a local file, honoring the env/arg/default precedence."""
    target_filename = filename or checkpoint.filename

    if weights is not None:
        return _resolve_user_source(
            os.fspath(weights), target_filename, revision, cache_dir
        )

    env_value = os.environ.get(checkpoint.weights_env_var)
    if env_value:
        return _resolve_user_source(env_value, target_filename, revision, cache_dir)

    path = _hf_download(
        checkpoint.repo_id, target_filename, revision or checkpoint.revision, cache_dir
    )
    return WeightSource(path=path, is_default=True)
