#!/usr/bin/env python3
"""Ask a Mitra checkpoint whether TabTune can load it.

    python -m tabtune.models.mitra.probe_mitra_v2
    python -m tabtune.models.mitra.probe_mitra_v2 autogluon/mitra-classifier-2
    python -m tabtune.models.mitra.probe_mitra_v2 /local/checkpoint/dir

With no arguments it probes every repo TabTune knows about: Mitra v1, Mitra v2
and the separate fine-tune checkpoint, for both tasks.

Exit status is 0 when every checkpoint probed is a plain swap, 2 when one needs
new architecture code, and 1 when one could not be fetched at all - which is a
network answer, not an architecture answer, and is reported as such.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

# Imported lazily inside the helpers below. Importing tabtune writes banner
# lines to STDOUT (colorlog's default stream), which would corrupt --json
# output before argparse has even run.

VERDICT_TEXT = {
    "checkpoint_swap": (
        "CHECKPOINT SWAP - the vendored Tab2D loaded it. Nothing to add; use it "
        "by model name."
    ),
    "needs_new_code": (
        "NEEDS NEW CODE - it was fetched but does not fit the vendored "
        "architecture. The error below names the mismatch."
    ),
    "unreachable": (
        "UNREACHABLE - it could not be fetched. This says NOTHING about the "
        "architecture; fix access and re-run before drawing a conclusion."
    ),
}


@contextlib.contextmanager
def _stdout_to_stderr():
    """Point file descriptor 1 at stderr for the duration of the block."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def targets(explicit: list[str]) -> list[str]:
    from tabtune.models.mitra.model_loading import MITRA_FINETUNE_REPO, MITRA_REPOS

    if explicit:
        return explicit
    seen, out = set(), []
    for variant in ("v1", "v2"):
        for repo in MITRA_REPOS[variant].values():
            if repo not in seen:
                seen.add(repo)
                out.append(repo)
    out.append(MITRA_FINETUNE_REPO)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", nargs="*",
                        help="repo id or local directory; default is all known repos")
    parser.add_argument("--device", default="cpu", help="device for the load test")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    # Under --json, stdout must carry the payload and nothing else. Importing
    # tabtune configures logging handlers that hold a direct reference to the
    # original stream, so rebinding sys.stdout does not reach them -- the
    # redirect has to happen at the file descriptor, below anything Python
    # holds a reference to.
    with _stdout_to_stderr() if args.json else contextlib.nullcontext():
        from tabtune.models.mitra.model_loading import probe_mitra_checkpoint

        results = [
            probe_mitra_checkpoint(c, device=args.device)
            for c in targets(args.checkpoint)
        ]

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for result in results:
            print(f"\n{result['source']}")
            print(f"  {VERDICT_TEXT[result['verdict']]}")
            if result["config"]:
                print(f"  config: {result['config']}")
            if result.get("n_parameters"):
                print(f"  parameters: {result['n_parameters']:,}")
            if result["unexpected_config_keys"]:
                print(f"  extra config keys (ignored by the loader, worth a look): "
                      f"{result['unexpected_config_keys']}")
            if result["error"]:
                print(f"  error: {result['error']}")

    verdicts = {r["verdict"] for r in results}
    if "needs_new_code" in verdicts:
        return 2
    if "unreachable" in verdicts:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
