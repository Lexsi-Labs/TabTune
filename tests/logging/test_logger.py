"""Run without model dependencies: pytest --confcutdir=tests/logging tests/logging."""

import asyncio
import builtins
import inspect
import io
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tabtune.logger import (
    get_logger,
    log_context,
    log_event,
    log_metrics,
    log_table,
    logged_operation,
    setup_logger,
    stage,
    track,
)


@pytest.fixture(autouse=True)
def isolate():
    logger = logging.getLogger("tabtune")
    saved = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    yield
    for handler in logger.handlers:
        handler.close()
    logger.handlers, logger.level, logger.propagate = saved


def records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_reconfigure_preserves_application_handlers_and_closes_owned(tmp_path):
    logger = logging.getLogger("tabtune")
    user = logging.NullHandler()
    logger.addHandler(user)
    setup_logger(log_file=tmp_path / "old.log", stream=io.StringIO())
    old = logger.handlers[-1]
    stream = io.StringIO()
    setup_logger(stream=stream)
    assert user in logger.handlers
    assert old.stream is None
    logger.info("exactly once")
    assert stream.getvalue().count("exactly once") == 1


def test_root_unchanged():
    root = logging.getLogger()
    old = root.handlers[:], root.level
    setup_logger(stream=io.StringIO())
    assert (root.handlers, root.level) == old


def test_json_wins_over_legacy_rich():
    stream = io.StringIO()
    setup_logger(use_rich=True, json_format=True, stream=stream)
    get_logger("TimeSeries.pipeline").info("hello")
    record = records(stream)[0]
    assert record["domain"] == "timeseries"
    assert record["schema_version"] == 1


def test_split_sinks_and_levels(tmp_path):
    stream = io.StringIO()
    path = tmp_path / "nested" / "run.jsonl"
    setup_logger(
        stream=stream,
        log_file=path,
        console_level="WARNING",
        file_level="DEBUG",
        file_format="json",
    )
    get_logger().debug("trace")
    get_logger().warning("careful")
    assert "trace" not in stream.getvalue()
    assert [r["level"] for r in records(io.StringIO(path.read_text()))] == ["DEBUG", "WARNING"]


def test_redaction_nonfinite_and_no_array_contents():
    import numpy as np

    stream = io.StringIO()
    setup_logger(json_format=True, stream=stream)
    with log_context(model="Naive", run_id="run-1"):
        log_event(
            get_logger(),
            "test",
            "[red]literal[/red]",
            config={"api_key": "secret", "nested": {"password": "bad"}},
            value=np.float64(float("nan")),
            data=np.array([91378]),
        )
    r = records(stream)[0]
    assert r["fields"]["config"]["api_key"] == "[REDACTED]"
    assert r["fields"]["value"] is None
    assert "91378" not in stream.getvalue()
    assert r["run_id"] == "run-1"


def test_stage_failure_and_context_restored():
    stream = io.StringIO()
    setup_logger(json_format=True, stream=stream)
    error = ValueError("bad schema")
    with pytest.raises(ValueError) as caught, stage(get_logger(), "fit", model="M"):
        raise error
    assert caught.value is error
    get_logger().info("after")
    data = records(stream)
    assert [r.get("event") for r in data] == ["stage_started", "stage_failed", None]
    assert data[1]["fields"]["elapsed_s"] >= 0
    assert "ValueError: bad schema" in data[1]["exception"]
    assert "run_id" not in data[2]


def test_nested_stage_shares_run_but_distinct_operation():
    stream = io.StringIO()
    setup_logger(json_format=True, stream=stream)
    with stage(get_logger(), "fit"):
        with stage(get_logger(), "load"):
            pass
    data = records(stream)
    assert len({r["run_id"] for r in data}) == 1
    assert len({r["operation_id"] for r in data}) == 2
    assert [r["operation"] for r in data] == ["fit", "load", "load", "fit"]


def test_context_isolated_across_threads():
    stream = io.StringIO()
    setup_logger(json_format=True, stream=stream)

    def worker(n):
        with log_context(run_id=str(n)):
            get_logger().info("worker %d", n)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, range(20)))
    assert all(r["message"].split()[-1] == r["run_id"] for r in records(stream))


def test_context_isolated_across_async_tasks():
    stream = io.StringIO()
    setup_logger(json_format=True, stream=stream)

    async def worker(n):
        with log_context(run_id=str(n)):
            await asyncio.sleep(0)
            get_logger().info("worker %d", n)

    async def main():
        await asyncio.gather(*(worker(i) for i in range(8)))

    asyncio.run(main())
    assert all(r["message"].split()[-1] == r["run_id"] for r in records(stream))


@pytest.mark.parametrize("rank,expected", [("0", True), ("1", False)])
def test_global_rank_wins(monkeypatch, rank, expected):
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("LOCAL_RANK", "0")
    stream = io.StringIO()
    setup_logger(stream=stream)
    get_logger().warning("warning")
    assert bool(stream.getvalue()) is expected


def test_rank_opt_out(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    stream = io.StringIO()
    setup_logger(stream=stream, rank_zero_only=False)
    get_logger().info("worker")
    assert "worker" in stream.getvalue()


def test_rich_markup_is_literal_and_no_color(monkeypatch):
    pytest.importorskip("rich")
    monkeypatch.setenv("NO_COLOR", "1")
    stream = io.StringIO()
    setup_logger(stream=stream, use_rich=True)
    get_logger("TimeSeries").warning("[red]literal[/red]")
    log_metrics(get_logger("TimeSeries"), {"rmse": 0.42})
    assert "[red]literal[/red]" in stream.getvalue()
    assert "TabTune-TS" in stream.getvalue()
    assert "\x1b" not in stream.getvalue()


def test_rich_missing_falls_back(monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.startswith("rich"):
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    stream = io.StringIO()
    setup_logger(stream=stream, use_rich=True)
    get_logger().info("fallback")
    assert "fallback" in stream.getvalue()


def test_table_truncation_and_json():
    stream = io.StringIO()
    setup_logger(stream=stream, json_format=True)
    log_table(get_logger(), "Results", ["name", "value"], [("a", 1), ("b", 2)], max_rows=1)
    assert records(stream)[0]["table"] == {
        "columns": ["name", "value"],
        "rows": [["a", 1]],
        "omitted": 1,
    }


def test_redirected_default_has_no_ansi():
    stream = io.StringIO()
    setup_logger(stream=stream)
    get_logger().info("hello")
    assert "\x1b" not in stream.getvalue()


def test_default_goes_to_stderr(capsys):
    setup_logger()
    get_logger().info("log")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "log" in captured.err


def test_rotation_is_utf8_and_bounded(tmp_path):
    path = tmp_path / "run.log"
    setup_logger(console=False, log_file=path, max_bytes=120, backup_count=2)
    for _ in range(20):
        get_logger().info("Forecast café")
    assert len(list(tmp_path.glob("run.log*"))) == 3
    assert "café" in path.read_text(encoding="utf-8")


def test_disabled_console_no_last_resort(capsys):
    setup_logger(console=False)
    get_logger().error("silent")
    assert capsys.readouterr().err == ""


def test_invalid_setup_preserves_previous_handler():
    stream = io.StringIO()
    setup_logger(stream=stream)
    with pytest.raises(ValueError):
        setup_logger(console_format="bad")
    get_logger().info("still works")
    assert "still works" in stream.getvalue()


def test_progress_does_not_claim_unfinished_work():
    stream = io.StringIO()
    setup_logger(stream=stream, json_format=True)
    with pytest.raises(ValueError):
        for i in track(range(3), logger=get_logger(), description="Work", min_interval=0):
            if i == 1:
                raise ValueError("stop")
    assert [r["fields"]["completed"] for r in records(stream)] == [1]


def test_decorator_preserves_signature_return_and_docstring():
    class Example:
        model_name = "demo"

        @logged_operation()
        def fit(self, x, *, flag=True):
            """Keep this docstring."""
            return x

    stream = io.StringIO()
    setup_logger(stream=stream, json_format=True)
    value = object()
    assert Example().fit(value) is value
    assert str(inspect.signature(Example.fit)) == "(self, x, *, flag=True)"
    assert Example.fit.__doc__ == "Keep this docstring."
    assert records(stream)[0]["model"] == "demo"


def test_import_is_lightweight():
    script = (
        "import sys,tabtune; assert 'torch' not in sys.modules; assert 'rich' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


def test_ts_baseline_end_to_end_and_json_stdout(capsys):
    from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema
    from tabtune.TimeSeries.data import make_panel, split_horizon

    data = make_panel(n_series=2, length=60)
    schema = TimeSeriesSchema(target="target", item_id="item_id", freq="h")
    train, test = split_horizon(data, schema, 4)
    stream = io.StringIO()
    setup_logger(stream=stream, json_format=True)
    pipe = TimeSeriesPipeline("Naive", forecast_params={"prediction_length": 4})
    assert pipe.fit(train, schema) is pipe
    assert pipe.predict().point.shape == (2, 4)
    result = pipe.evaluate(test, output_format="json")
    assert json.loads(capsys.readouterr().out)["mae"] == result["mae"]
    backtest = pipe.backtest(windows=2, output_format="rich")
    assert len(backtest) == 2
    assert any(r.get("event") == "table" for r in records(stream))
    assert all(r["domain"] == "timeseries" for r in records(stream))


def test_tabular_evaluate_regression_branch_without_backend_imports(capsys):
    # Execute the real method body with lightweight collaborators. Vendored
    # model imports are irrelevant to testing the report's branching logic.
    import ast

    import numpy as np

    source = Path(__file__).parents[2] / "tabtune" / "TabularPipeline" / "pipeline.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TabularPipeline")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    fn.decorator_list = []
    env = {"pd": object(), "logger": get_logger(), "json": json, "log_metrics": log_metrics}
    # Annotations are deferred for this isolated method test.
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            fn,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), env)

    class Regression:
        _is_fitted = True
        task_type = "regression"

        def predict(self, x):
            return np.array([1.0, 2.0])

        def _calculate_regression_metrics(self, y, p):
            return {"rmse": 0.5}

    stream = io.StringIO()
    setup_logger(stream=stream)
    result = env["evaluate"](Regression(), None, None, "rich")
    assert result == {"rmse": 0.5}
    assert "Unknown output_format" not in stream.getvalue()
    assert "rmse" in stream.getvalue()


def test_rich_progress_ends_before_next_record():
    pytest.importorskip("rich")
    stream = io.StringIO()
    setup_logger(stream=stream, use_rich=True)
    log_event(get_logger(), "progress", "Processed", completed=2, total=2)
    get_logger().info("next record")
    lines = stream.getvalue().splitlines()
    assert not any("━" in line and "next record" in line for line in lines)


def test_metrics_are_machine_readable():
    stream = io.StringIO()
    setup_logger(stream=stream, json_format=True)
    log_metrics(get_logger(), {"auc": 0.9})
    assert records(stream)[0]["metrics"] == {"auc": 0.9}
    assert records(stream)[0]["event"] == "metrics"


def test_nested_failure_has_one_traceback():
    stream = io.StringIO()
    setup_logger(stream=stream, json_format=True)
    with pytest.raises(ValueError), stage(get_logger(), "fit"):
        with stage(get_logger(), "load"):
            raise ValueError("broken checkpoint")
    data = records(stream)
    assert sum("exception" in r for r in data) == 1
    assert data[-1]["operation"] == "fit"
