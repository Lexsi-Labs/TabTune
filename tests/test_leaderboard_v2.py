"""Tests for the overhauled leaderboard.

The leaderboard is exercised against a stub pipeline rather than real
checkpoints, so these tests run in milliseconds and gate every pull request.
What they verify is the leaderboard's own logic: ranking, failure isolation,
Pareto computation and export - all of which had real defects before 0.2.0.
"""

from __future__ import annotations

import json
import time
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from tabtune.TabularLeaderboard import LeaderboardEntry, TabularLeaderboard

# Resolved via import_module rather than `import a.b.c as x`: under pytest's
# importlib import mode the latter can resolve against a shadowing top-level
# namespace package when the full suite is collected together.
lb_module = import_module("tabtune.TabularLeaderboard.leaderboard")

pytestmark = [pytest.mark.unit, pytest.mark.leaderboard]


# ------------------------------------------------------------------- fixtures


class _StubPipeline:
    """Minimal stand-in for TabularPipeline, parameterised by model name.

    ``_Broken`` raises on fit so failure handling can be tested without needing
    a genuinely broken checkpoint.
    """

    #: model name -> (accuracy-ish skill in [0, 1], simulated predict latency)
    BEHAVIOUR = {
        "TabICLv2": (0.95, 0.05),
        "Mitra": (0.90, 0.01),
        "OrionMSP": (0.85, 0.20),
        "TabPFNv3": (0.98, 0.50),
        "_Broken": (0.0, 0.0),
    }

    def __init__(self, model_name, task_type="classification", **kwargs):
        self.model_name = model_name
        self.task_type = task_type
        self.skill, self.latency = self.BEHAVIOUR.get(model_name, (0.5, 0.1))
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        if self.model_name == "_Broken":
            raise RuntimeError("simulated checkpoint failure")
        self._y = np.asarray(y)
        return self

    def predict(self, X):
        # Sleep proportionally so `predict_seconds` is genuinely ordered; the
        # Pareto computation reads measured wall-clock, not this table.
        time.sleep(self.latency * 0.1)
        # Deterministically get `skill` fraction of the labels right.
        n = len(X)
        truth = self._truth(n)
        correct = int(round(self.skill * n))
        out = truth.copy()
        out[correct:] = 1 - out[correct:]
        return out

    def predict_proba(self, X):
        preds = self.predict(X)
        confidence = 0.5 + 0.45 * self.skill
        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds] = confidence
        proba[np.arange(len(preds)), 1 - preds] = 1 - confidence
        return proba

    def _truth(self, n):
        return np.array([i % 2 for i in range(n)])


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace TabularPipeline inside the leaderboard module with the stub."""
    monkeypatch.setattr(lb_module, "TabularPipeline", _StubPipeline)
    return _StubPipeline


@pytest.fixture
def split():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    X_test = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
    y_train = pd.Series([i % 2 for i in range(60)])
    y_test = pd.Series([i % 2 for i in range(40)])
    return X_train, X_test, y_train, y_test


@pytest.fixture
def board(split, stub_pipeline):
    return TabularLeaderboard(*split)


# --------------------------------------------------------------- construction


def test_rejects_bad_task_type(split):
    with pytest.raises(ValueError, match="task_type"):
        TabularLeaderboard(*split, task_type="clustering")


def test_rejects_mismatched_split_lengths(split):
    X_train, X_test, y_train, y_test = split
    with pytest.raises(ValueError, match="y_train"):
        TabularLeaderboard(X_train, X_test, y_train.iloc[:-1], y_test)


def test_add_model_chains_and_counts(board):
    assert board.add_model("TabICLv2").add_model("Mitra") is board
    assert len(board) == 2


def test_add_models_bulk(board):
    board.add_models(["TabICLv2", "Mitra", "OrionMSP"])
    assert len(board) == 3


def test_add_all_uses_the_registry(board):
    board.add_all(commercial_ok=True)
    names = {c["model_name"] for c in board.models_to_run}
    assert "Mitra" in names
    assert "TabPFNv3" not in names  # research-only weights


# ---------------------------------------------------------------------- run


def test_run_reports_full_metric_bundle(board):
    board.add_model("TabICLv2")
    frame = board.run(display=False)
    for column in ("accuracy", "f1_score", "roc_auc_score", "ece", "brier_score"):
        assert column in frame.columns
    assert "fit_s" in frame.columns and "predict_s" in frame.columns


def test_run_records_licenses(board):
    board.add_models(["TabICLv2", "TabPFNv3"])
    frame = board.run(display=False)
    assert set(frame["Commercial"]) <= {"yes", "no", "unverified"}
    assert frame.loc[frame["Model"] == "TabPFNv3", "Commercial"].iloc[0] == "no"


def test_ranking_puts_the_best_model_first(board):
    board.add_models(["OrionMSP", "TabPFNv3", "Mitra"])
    frame = board.run(rank_by="accuracy", display=False)
    assert frame.iloc[0]["Model"] == "TabPFNv3"
    assert frame.index.name == "Rank"
    assert list(frame.index) == [1, 2, 3]


def test_ranking_on_a_lower_is_better_metric_inverts(board):
    board.add_models(["OrionMSP", "TabPFNv3"])
    frame = board.run(rank_by="ece", display=False)
    values = frame["ece"].tolist()
    assert values == sorted(values)


def test_unknown_rank_metric_warns_and_falls_back(board):
    board.add_model("Mitra")
    with pytest.warns(UserWarning, match="not a metric"):
        board.run(rank_by="vibes", display=False)
    assert board._rank_by == "roc_auc_score"


def test_legacy_display_names_still_rank(board):
    board.add_models(["Mitra", "OrionMSP"])
    board.run(rank_by="ROC AUC", display=False)
    assert board._rank_by == "roc_auc_score"


# --------------------------------------------------------------- failures


def test_a_failing_model_does_not_abort_the_run(board):
    board.add_models(["TabICLv2", "_Broken", "Mitra"])
    frame = board.run(display=False)
    assert len(frame) == 3
    assert (frame["Status"] == "failed").sum() == 1


def test_failed_entries_hold_nan_not_the_string_failed(board):
    """A text sentinel in a numeric column silently corrupts every later sort."""
    board.add_models(["TabICLv2", "_Broken"])
    frame = board.run(display=False)
    assert pd.api.types.is_numeric_dtype(frame["accuracy"])
    failed = frame[frame["Status"] == "failed"]
    assert failed["accuracy"].isna().all()


def test_failed_entries_sort_last_regardless_of_direction(board):
    board.add_models(["_Broken", "TabICLv2"])
    board.run(display=False)
    for metric in ("accuracy", "ece"):
        frame = board.to_frame(rank_by=metric)
        assert frame.iloc[-1]["Status"] == "failed"


def test_error_column_is_dropped_when_nothing_fails(board):
    board.add_model("Mitra")
    assert "Error" not in board.run(display=False).columns


def test_error_text_and_traceback_are_preserved(board):
    board.add_model("_Broken")
    board.run(display=False)
    entry = board.entries[0]
    assert not entry.ok
    assert "simulated checkpoint failure" in entry.error
    assert entry.traceback and "RuntimeError" in entry.traceback


# ----------------------------------------------------------------- analysis


def test_best_returns_the_top_entry(board):
    board.add_models(["OrionMSP", "TabPFNv3"])
    board.run(rank_by="accuracy", display=False)
    assert board.best().model_name == "TabPFNv3"


def test_best_ignores_failures(board):
    board.add_models(["_Broken", "Mitra"])
    board.run(display=False)
    assert board.best().model_name == "Mitra"


def test_best_is_none_when_everything_fails(board):
    board.add_model("_Broken")
    board.run(display=False)
    assert board.best() is None


def test_pareto_front_excludes_dominated_entries(board):
    # Mitra: fast and decent. TabPFNv3: slow and best. OrionMSP: slow and worst,
    # so it is dominated by both and must not appear.
    board.add_models(["Mitra", "TabPFNv3", "OrionMSP"])
    board.run(rank_by="accuracy", display=False)
    names = {e.model_name for e in board.pareto_front("accuracy")}
    assert "OrionMSP" not in names
    assert "Mitra" in names


def test_pareto_front_is_ordered_by_latency(board):
    board.add_models(["Mitra", "TabPFNv3", "OrionMSP"])
    board.run(rank_by="accuracy", display=False)
    front = board.pareto_front("accuracy")
    latencies = [e.predict_seconds for e in front]
    assert latencies == sorted(latencies)


def test_to_frame_reranks_without_rerunning(board):
    board.add_models(["Mitra", "TabPFNv3"])
    board.run(rank_by="accuracy", display=False)
    n_entries = len(board.entries)
    reranked = board.to_frame(rank_by="ece")
    assert len(board.entries) == n_entries
    assert len(reranked) == 2


# ------------------------------------------------------------------ exports


def test_to_markdown(board, tmp_path):
    board.add_model("Mitra")
    board.run(display=False)
    path = tmp_path / "lb.md"
    text = board.to_markdown(path)
    assert "Mitra" in text and path.exists()


def test_to_csv_round_trips(board, tmp_path):
    board.add_models(["Mitra", "TabICLv2"])
    board.run(display=False)
    path = board.to_csv(tmp_path / "lb.csv")
    assert len(pd.read_csv(path)) == 2


def test_to_json_is_valid_and_complete(board, tmp_path):
    board.add_models(["Mitra", "_Broken"])
    board.run(display=False)
    path = board.to_json(tmp_path / "lb.json")
    payload = json.loads(path.read_text())
    assert payload["task_type"] == "classification"
    assert len(payload["entries"]) == 2
    assert any(e["status"] == "failed" for e in payload["entries"])


def test_to_html_is_self_contained(board, tmp_path):
    board.add_models(["Mitra", "TabPFNv3", "OrionMSP"])
    board.run(rank_by="accuracy", display=False)
    path = tmp_path / "lb.html"
    html = board.to_html(path)

    assert path.exists()
    assert html.startswith("<!DOCTYPE html>")
    # Self-contained: no external stylesheets, scripts or images.
    for forbidden in ("<script", "src=\"http", "href=\"http"):
        assert forbidden not in html
    assert "<svg" in html  # the Pareto chart rendered
    assert "TabPFNv3" in html


def test_html_escapes_untrusted_content(split, stub_pipeline, monkeypatch):
    board = TabularLeaderboard(*split)
    board.add_model("Mitra")
    board.run(display=False)
    board.entries[0].error = "<script>alert(1)</script>"
    board.entries[0].model_name = "<img onerror=x>"
    board._ranked = board._build_frame("accuracy")
    html = board.to_html()
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html or "&lt;img" in html


def test_html_without_pareto(board):
    board.add_model("Mitra")
    board.run(display=False)
    assert "<svg" not in board.to_html(include_pareto=False)


def test_summary_mentions_the_deployable_alternative(board):
    board.add_models(["TabPFNv3", "Mitra"])
    board.run(rank_by="accuracy", display=False)
    text = board.summary()
    assert "Best:" in text
    assert "commercially deployable" in text.lower()


def test_summary_before_running(split, stub_pipeline):
    assert "not been run" in TabularLeaderboard(*split).summary()


def test_results_is_empty_before_running(split, stub_pipeline):
    assert TabularLeaderboard(*split).results.empty


def test_progress_callback_is_invoked(board):
    seen = []
    board.add_models(["Mitra", "TabICLv2"])
    board.run(display=False, progress=lambda i, n, name: seen.append((i, n, name)))
    assert seen == [(1, 2, "Mitra"), (2, 2, "TabICLv2")]


# ------------------------------------------------------------------- entries


def test_entry_display_name_includes_mode():
    entry = LeaderboardEntry("TabICL", "finetune", "classification", finetune_mode="sft")
    assert entry.display_name == "TabICL / finetune / sft"


def test_entry_label_overrides_display_name():
    entry = LeaderboardEntry("TabICL", "inference", "classification", label="baseline")
    assert entry.display_name == "baseline"


@pytest.mark.parametrize(
    "flag,expected", [(True, "yes"), (False, "no"), (None, "unverified")]
)
def test_entry_commercial_badge(flag, expected):
    entry = LeaderboardEntry("X", "inference", "classification", commercial_use_ok=flag)
    assert entry.commercial_badge == expected


def test_leaderboard_module_imports_without_ipython(monkeypatch):
    """The module used to import IPython at top level, breaking headless use."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    module = importlib.reload(lb_module)
    assert hasattr(module, "TabularLeaderboard")
