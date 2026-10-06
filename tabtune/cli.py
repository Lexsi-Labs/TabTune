"""``tabtune`` command line entry point.

Command groups load lazily so the CLI starts fast:

* ``tabtune timeseries ...``: time series models (:mod:`tabtune.TimeSeries.cli`)
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

__all__ = ["main"]

_GROUPS = {
    "timeseries": (
        "time series models: list-models, info, forecast, evaluate, anomalies, "
        "impute, embed, benchmark"
    )
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("usage: tabtune <group> <command> [options]\n\ngroups:")
        for name, text in _GROUPS.items():
            print(f"  {name:<11} {text}")
        return 0
    group, rest = args[0], args[1:]
    if group == "timeseries":
        from .TimeSeries.cli import main as timeseries_main

        return timeseries_main(rest)
    print(f"unknown command group {group!r}; known: {list(_GROUPS)}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
