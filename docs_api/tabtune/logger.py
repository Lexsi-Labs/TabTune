"""Shared console presentation and structured logging for TabTune and TabTune-TS.

No ML dependencies are imported here. Existing ``logging.getLogger`` calls and
``setup_logger`` arguments remain supported; Rich is loaded only when selected.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "setup_logger",
    "get_logger",
    "log_banner",
    "log_context",
    "log_event",
    "log_table",
    "log_metrics",
    "stage",
    "logged_operation",
    "track",
]

_CONTEXT = contextvars.ContextVar("tabtune_log_context", default=MappingProxyType({}))
_LOCK = threading.RLock()
_STAGE_DEPTH = contextvars.ContextVar("tabtune_stage_depth", default=0)
_SECRET = re.compile(r"(?:password|passwd|secret|token|api[_-]?key|authorization|credential)", re.I)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PREFIX = re.compile(
    r"^\[(?:Pipeline|TimeSeriesPipeline|Leaderboard|TimeSeriesLeaderboard|TuningManager|DataProcessor|TimeSeriesBenchmark|TimeSeriesEnsemble)\]\s*"
)
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_STYLES = {
    "DEBUG": "dim",
    "INFO": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
}


_BANNER = r"""
  ████████╗ █████╗ ██████╗  ████████╗██╗   ██╗███╗   ██╗███████╗
  ╚══██╔══╝██╔══██╗██╔══██╗ ╚══██╔══╝██║   ██║████╗  ██║██╔════╝
     ██║   ███████║██████╔╝    ██║   ██║   ██║██╔██╗ ██║█████╗
     ██║   ██╔══██║██╔══██╗    ██║   ██║   ██║██║╚██╗██║██╔══╝
     ██║   ██║  ██║██████╔╝    ██║   ╚██████╔╝██║ ╚████║███████╗
     ╚═╝   ╚═╝  ╚═╝╚═════╝     ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝
""".strip("\n")

_TS_SUFFIX = (
    "     ████████╗███████╗",
    "     ╚══██╔══╝██╔════╝",
    "  ━━    ██║   ███████╗",
    "        ██║   ╚════██║",
    "        ██║   ███████║",
    "        ╚═╝   ╚══════╝",
)
_BANNER_TS = "\n".join(
    line.ljust(max(map(len, _BANNER.splitlines()))) + suffix
    for line, suffix in zip(_BANNER.splitlines(), _TS_SUFFIX, strict=True)
)
_BANNER_PRODUCTS = {
    "tabular": {
        "name": "TabTune",
        "art": _BANNER,
        "tagline": "Unified Library for Fine-Tuning and Inference of Foundational Tabular Models",
    },
    "timeseries": {
        "name": "TabTune-TS",
        "art": _BANNER_TS,
        "tagline": "Forecasting · Fine-Tuning · Backtesting · Calibration",
    },
}
_BANNERS_SHOWN = set()


def _clean(value):
    return _ANSI.sub("", str(value)).replace("\r", "\\r").replace("\x1b", "")


def _safe(value, depth=0):
    """Convert metadata without serialising dataset/tensor contents or reprs."""
    if depth > 6:
        return "<nested>"
    if value is None or isinstance(value, (bool, int, str)):
        return _clean(value) if isinstance(value, str) else value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(k): "[REDACTED]" if _SECRET.search(str(k)) else _safe(v, depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_safe(v, depth + 1) for v in value]
    if isinstance(value, Path):
        return _clean(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if getattr(value, "shape", None) == () and hasattr(value, "item"):
        return _safe(value.item(), depth + 1)
    return f"<{type(value).__name__}>"


def _display(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, allow_nan=False)
    return _clean(value)


def _domain(name):
    return "timeseries" if "timeseries" in name.lower() else "tabular"


def _brand(record):
    return "TabTune-TS" if record.domain == "timeseries" else "TabTune"


class _ContextFilter(logging.Filter):
    def filter(self, record):
        for key, value in _CONTEXT.get().items():
            if key not in _RESERVED and not hasattr(record, key):
                setattr(record, key, value)
        if not hasattr(record, "domain"):
            record.domain = _domain(record.name)
        return True


def _rank_zero():
    # Global rank wins: LOCAL_RANK=0 is not necessarily the global leader.
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return rank in ("", "0")


class _RankFilter(logging.Filter):
    def filter(self, record):
        return _rank_zero()


class _BannerFilter(logging.Filter):
    def __init__(self, enabled):
        super().__init__()
        self.enabled = enabled

    def filter(self, record):
        return self.enabled or getattr(record, "event", None) != "banner"


def _banner_has_sink(logger):
    """Respect owned output settings before application capture handlers."""
    handlers = []
    current = logger
    while current is not None:
        handlers.extend(h for h in current.handlers if not isinstance(h, logging.NullHandler))
        if not current.propagate:
            break
        current = current.parent
    owned = [h for h in handlers if getattr(h, "_tabtune_owned", False)]
    for handler in owned or handlers:
        if handler.level > logging.INFO:
            continue
        if any(isinstance(f, _BannerFilter) and not f.enabled for f in handler.filters):
            continue
        if not _rank_zero() and any(isinstance(f, _RankFilter) for f in handler.filters):
            continue
        return True
    return False


def log_banner(logger, *, domain=None):
    """Log each product's banner once per process, when INFO can reach a sink.

    The artwork is one structured logging event, never a direct stdout print.
    Reconfiguring logging does not reset the per-product once-only state.
    """
    domain = domain or _domain(logger.name)
    if domain not in _BANNER_PRODUCTS:
        raise ValueError("banner domain must be 'tabular' or 'timeseries'")
    if not logger.isEnabledFor(logging.INFO):
        return
    with _LOCK:
        if domain in _BANNERS_SHOWN or not _banner_has_sink(logger):
            return
        banner = {**_BANNER_PRODUCTS[domain]}
        banner["version"] = str(getattr(sys.modules.get("tabtune"), "__version__", "unknown"))
        _BANNERS_SHOWN.add(domain)
        logger.info(
            "%s initialized",
            banner["name"],
            extra={"event": "banner", "domain": domain, "banner": banner},
        )


def _payload(record):
    payload = {
        "schema_version": 1,
        "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "level": record.levelname,
        "logger": record.name,
        "message": _clean(record.getMessage()),
    }
    payload.update(
        _safe(
            {
                k: v
                for k, v in record.__dict__.items()
                if k not in _RESERVED and not k.startswith("_")
            }
        )
    )
    if record.exc_info:
        payload["exception"] = _clean(logging.Formatter().formatException(record.exc_info))
    if record.stack_info:
        payload["stack"] = _clean(record.stack_info)
    return payload


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(_payload(record), ensure_ascii=True, allow_nan=False)


class _PlainFormatter(logging.Formatter):
    def format(self, record):
        banner = getattr(record, "banner", None)
        if getattr(record, "event", None) == "banner" and banner:
            return (
                "\n"
                + _clean(banner["art"])
                + "\n\n"
                + (f"{banner['name']} / v{banner['version']}\n{banner['tagline']}\n")
            )
        clock = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        message = _PREFIX.sub("", _clean(record.getMessage())).strip()
        fields = _safe(getattr(record, "fields", {}))
        context = " ".join(
            f"{k}={_display(_safe(getattr(record, k)))}"
            for k in ("model", "operation", "run_id")
            if hasattr(record, k)
        )
        tail = " ".join(f"{k}={_display(v)}" for k, v in fields.items())
        result = f"{clock} {record.levelname:<7} {_brand(record)} | {context + ' | ' if context else ''}{message}"
        if tail:
            result += " | " + tail
        table = _safe(getattr(record, "table", None))
        if table:
            result += "\n  " + " | ".join(map(str, table["columns"]))
            result += "".join(
                "\n  " + " | ".join(_display(v) for v in row) for row in table["rows"]
            )
            if table["omitted"]:
                result += f"\n  ... {table['omitted']} more rows"
        if record.exc_info:
            result += "\n" + _clean(self.formatException(record.exc_info))
        if record.stack_info:
            result += "\n" + _clean(record.stack_info)
        return result


class _RichHandler(logging.Handler):
    """Render records as literal text; dataset names are never Rich markup."""

    def __init__(self, console):
        super().__init__()
        self.console = console

    def emit(self, record):
        try:
            from rich import box
            from rich.panel import Panel
            from rich.progress_bar import ProgressBar
            from rich.table import Table
            from rich.text import Text

            domain_style = "magenta" if record.domain == "timeseries" else "cyan"
            banner = getattr(record, "banner", None)
            if getattr(record, "event", None) == "banner" and banner:
                art = banner["art"]
                if self.console.width < max(map(len, art.splitlines())) + 6:
                    art = (
                        _BANNER
                        if self.console.width >= max(map(len, _BANNER.splitlines())) + 6
                        else banner["name"]
                    )
                content = Text()
                content.append(art, style=f"bold {domain_style}")
                content.append("\n\n" + banner["tagline"])
                self.console.print(
                    Panel(
                        content,
                        box=box.ROUNDED,
                        border_style=domain_style,
                        expand=False,
                        padding=(1, 2),
                        title=Text(f"{banner['name']} / v{banner['version']}"),
                    )
                )
                return
            clock = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            line = Text()
            line.append(clock + "  ", style="dim")
            line.append(f"{record.levelname:<7} ", style=_STYLES.get(record.levelname, ""))
            line.append(_brand(record), style=domain_style)
            for key in ("model", "operation"):
                if hasattr(record, key):
                    line.append(" / " + _display(_safe(getattr(record, key))), style="dim")
            message = _PREFIX.sub("", _clean(record.getMessage())).strip()
            line.append(
                "  " + message,
                style="green" if getattr(record, "event", None) == "stage_completed" else "",
            )
            self.console.print(line)
            fields = _safe(getattr(record, "fields", {}))
            if fields:
                detail = Text("  ")
                for i, (key, value) in enumerate(fields.items()):
                    detail.append(("   " if i else "") + f"{key}=", style="dim")
                    detail.append(_display(value))
                self.console.print(detail)
            table = _safe(getattr(record, "table", None))
            if table:
                grid = Table(box=box.SIMPLE_HEAD, border_style="dim", padding=(0, 1), expand=False)
                for i, column in enumerate(table["columns"]):
                    numeric = bool(table["rows"]) and all(
                        i < len(row) and (row[i] is None or isinstance(row[i], (int, float)))
                        for row in table["rows"]
                    )
                    grid.add_column(
                        Text(str(column), style=domain_style),
                        justify="right" if numeric else "left",
                    )
                for row in table["rows"]:
                    grid.add_row(*(Text(_display(v)) for v in row))
                self.console.print(grid)
                if table["omitted"]:
                    self.console.print(Text(f"  ... {table['omitted']} more rows", style="dim"))
            if getattr(record, "event", None) == "progress" and fields.get("total"):
                self.console.print(
                    ProgressBar(
                        total=fields["total"],
                        completed=fields["completed"],
                        width=min(44, max(10, self.console.width - 4)),
                        complete_style=domain_style,
                        finished_style=domain_style,
                    )
                )
                self.console.print()
            if record.exc_info:
                # Never expose local variables through enhanced tracebacks.
                self.console.print(
                    Text(_clean(logging.Formatter().formatException(record.exc_info)), style="red")
                )
            if record.stack_info:
                self.console.print(Text(_clean(record.stack_info), style="dim"))
        except Exception:
            self.handleError(record)


def _interactive(stream):
    if getattr(stream, "isatty", lambda: False)():
        return True
    return "ipykernel" in sys.modules


def setup_logger(
    level=logging.INFO,
    log_file=None,
    use_color=True,
    use_rich=None,
    json_format=False,
    max_bytes=5 * 1024 * 1024,
    backup_count=5,
    *,
    console_format=None,
    file_format=None,
    console_level=None,
    file_level=None,
    stream=None,
    console=True,
    propagate=False,
    rank_zero_only=True,
    show_banner=True,
):
    """Configure owned handlers, leaving application handlers intact.

    ``console_format`` is auto/rich/plain/json; ``file_format`` is plain/json.
    Logs go to stderr by default. ``json_format=True`` selects JSON on both
    sinks and takes precedence over legacy ``use_rich``. Explicit format
    arguments take precedence over legacy switches. Files use UTF-8 rotation.
    Use distinct log files per process, or the default rank-zero filtering.
    ``show_banner=False`` suppresses product banners on owned sinks.
    """
    if max_bytes < 0 or backup_count < 0:
        raise ValueError("max_bytes and backup_count must be non-negative")
    stream = sys.stderr if stream is None else stream
    fmt = console_format or (
        "json"
        if json_format
        else "rich"
        if use_rich is True
        else "plain"
        if use_rich is False
        else "auto"
    )
    file_fmt = file_format or ("json" if json_format else "plain")
    if fmt not in {"auto", "rich", "plain", "json"} or file_fmt not in {"plain", "json"}:
        raise ValueError(
            "console_format must be auto/rich/plain/json; file_format must be plain/json"
        )
    level = logging._checkLevel(level)
    console_level = logging._checkLevel(level if console_level is None else console_level)
    file_level = logging._checkLevel(level if file_level is None else file_level)
    handlers = []
    try:
        if console:
            rich_console = None
            if fmt == "rich" or (fmt == "auto" and _interactive(stream)):
                try:
                    from rich.console import Console

                    rich_console = Console(
                        file=stream,
                        no_color=not use_color or "NO_COLOR" in os.environ,
                        highlight=False,
                    )
                except ImportError:
                    pass
            if rich_console is not None:
                handler = _RichHandler(rich_console)
            else:
                handler = logging.StreamHandler(stream)
                handler.setFormatter(_JsonFormatter() if fmt == "json" else _PlainFormatter())
            handler.setLevel(console_level)
            handlers.append(handler)
        if log_file:
            path = Path(log_file).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            handlers.append(handler)
            handler.setLevel(file_level)
            handler.setFormatter(_JsonFormatter() if file_fmt == "json" else _PlainFormatter())
        if not handlers:
            handlers.append(logging.NullHandler())
        for handler in handlers:
            handler._tabtune_owned = True
            handler.addFilter(_ContextFilter())
            handler.addFilter(_BannerFilter(show_banner))
            if rank_zero_only:
                handler.addFilter(_RankFilter())
    except Exception:
        for handler in handlers:
            handler.close()
        raise
    logger = logging.getLogger("tabtune")
    with _LOCK:
        old = [h for h in logger.handlers if getattr(h, "_tabtune_owned", False)]
        for handler in old:
            logger.removeHandler(handler)
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(
            min([level] + [h.level for h in handlers if not isinstance(h, logging.NullHandler)])
        )
        logger.propagate = propagate
        for handler in old:
            handler.close()
    return logger


def get_logger(name="tabtune"):
    """Return a standard logger beneath the shared TabTune namespace."""
    return logging.getLogger(
        name if name == "tabtune" or name.startswith("tabtune.") else f"tabtune.{name}"
    )


@contextmanager
def log_context(**fields):
    """Bind metadata to this thread/async context; always restore the caller."""
    if _RESERVED.intersection(fields):
        raise ValueError("context keys cannot shadow LogRecord attributes")
    token = _CONTEXT.set({**_CONTEXT.get(), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def log_event(logger, event, message, *, level=logging.INFO, exc_info=None, **fields):
    if logger.isEnabledFor(level):
        extra = {**_CONTEXT.get(), "event": event, "fields": _safe(fields)}
        logger.log(level, message, extra=extra, exc_info=exc_info, stacklevel=2)


def log_table(logger, title, columns, rows, *, max_rows=20, level=logging.INFO, metrics=None):
    """Emit one structured table; bound console/file size without changing results."""
    if not logger.isEnabledFor(level):
        return
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    rows = list(rows)
    table = _safe(
        {"columns": list(columns), "rows": rows[:max_rows], "omitted": max(0, len(rows) - max_rows)}
    )
    logger.log(
        level,
        title,
        extra={
            **_CONTEXT.get(),
            "event": "metrics" if metrics is not None else "table",
            "table": table,
            **({"metrics": _safe(metrics)} if metrics is not None else {}),
        },
        stacklevel=2,
    )


def log_metrics(logger, metrics, *, title="Evaluation", labels=None):
    labels = labels or {}
    log_table(
        logger,
        title,
        ["Metric", "Value"],
        [(labels.get(k, k), v) for k, v in metrics.items() if not isinstance(v, Mapping)],
        metrics=metrics,
    )


@contextmanager
def stage(logger, name, *, level=logging.INFO, **fields):
    """Time a real operation and preserve exceptions, including cancellation."""
    parent = _CONTEXT.get()
    start = time.perf_counter()
    with log_context(
        run_id=parent.get("run_id", uuid.uuid4().hex[:12]),
        operation_id=uuid.uuid4().hex[:12],
        operation=name,
        **fields,
    ):
        log_event(logger, "stage_started", f"{name.capitalize()} started", level=level)
        depth = _STAGE_DEPTH.get()
        depth_token = _STAGE_DEPTH.set(depth + 1)
        try:
            yield
        except BaseException as exc:
            cancelled = isinstance(exc, (KeyboardInterrupt, SystemExit))
            log_event(
                logger,
                "stage_cancelled" if cancelled else "stage_failed",
                f"{name.capitalize()} {'cancelled' if cancelled else 'failed'}",
                level=logging.WARNING if cancelled else logging.ERROR,
                elapsed_s=round(time.perf_counter() - start, 6),
                error_type=type(exc).__name__,
                exc_info=not cancelled and depth == 0,
            )
            raise
        else:
            log_event(
                logger,
                "stage_completed",
                f"{name.capitalize()} complete",
                level=level,
                elapsed_s=round(time.perf_counter() - start, 6),
            )
        finally:
            _STAGE_DEPTH.reset(depth_token)


def logged_operation(name=None):
    """Instrument pipeline methods without changing signatures, results or state."""

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            logger = get_logger(fn.__module__)
            operation = name or fn.__name__
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            level = (
                logging.DEBUG
                if _CONTEXT.get().get("operation")
                or ("output_format" in bound.arguments and bound.arguments["output_format"] is None)
                else logging.INFO
            )
            metadata = {"domain": _domain(fn.__module__)}
            for attr, key in (
                ("model_name", "model"),
                ("task_type", "task"),
                ("tuning_strategy", "strategy"),
            ):
                if hasattr(self, attr):
                    metadata[key] = getattr(self, attr)
            with log_context(**metadata), stage(logger, operation, level=level):
                return fn(self, *args, **kwargs)

        return wrapped

    return decorate


def track(iterable, *, logger, description, total=None, min_interval=1.0):
    """Log completed work at most once per interval, plus final state.

    Progress advances only after the loop body succeeds. No background threads,
    cursor rewriting, or guessed ETA; works identically in notebooks and CI.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            pass
    start = last = time.perf_counter()
    completed = 0
    for item in iterable:
        yield item
        completed += 1
        now = time.perf_counter()
        if now - last >= min_interval and completed != total:
            log_event(
                logger,
                "progress",
                description,
                completed=completed,
                total=total,
                elapsed_s=round(now - start, 3),
            )
            last = now
    log_event(
        logger,
        "progress",
        description,
        completed=completed,
        total=total,
        elapsed_s=round(time.perf_counter() - start, 3),
    )
