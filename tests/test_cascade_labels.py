"""CascadeStackingEnsemble maps classification labels to probability columns (issue #36)."""
from __future__ import annotations

import numpy as np
import pytest

from tabtune.ensemble.strategies import CascadeStackingEnsemble

pytestmark = pytest.mark.unit


def _one_hot(cols, n=3):
    return np.eye(n)[cols] * 0.9 + 0.1 / n


# Validation labels lack class 1; "good" predicts them, "wrong" predicts class 1 instead of 2.
_COLS = [0, 0, 2, 2]
_GOOD, _WRONG = _one_hot([0, 0, 2, 2]), _one_hot([0, 0, 1, 1])


def _weights(y_val, **kw):
    ens = CascadeStackingEnsemble(n_levels=1, ges_size=5)
    ens.fit([{"good": _GOOD, "wrong": _WRONG}], y_val, **kw)
    return ens.weights_


def test_encoded_labels_pick_the_correct_block_even_when_a_class_is_missing():
    assert _weights(np.array(_COLS)) == {"L1_good": 1.0, "L1_wrong": 0.0}


def test_original_labels_with_classes_match_encoded_labels():
    classes = np.array(["cat", "dog", "eel"])
    assert _weights(classes[_COLS], classes=classes) == {"L1_good": 1.0, "L1_wrong": 0.0}
    ints = np.array([10, 20, 30])
    assert _weights(ints[_COLS], classes=ints) == {"L1_good": 1.0, "L1_wrong": 0.0}


@pytest.mark.parametrize("y_val", [np.array(["cat", "cat", "eel", "eel"]), np.array([1, 1, 3, 3])])
def test_labels_that_are_not_column_indices_raise_without_classes(y_val):
    with pytest.raises(ValueError, match="classes="):
        _weights(y_val)


def test_classes_must_cover_y_val_and_match_the_columns():
    with pytest.raises(ValueError, match="not in classes"):
        _weights(np.array(["cat", "cat", "fox", "fox"]), classes=np.array(["cat", "dog", "eel"]))
    with pytest.raises(ValueError, match="columns"):
        _weights(np.array(_COLS), classes=np.array([0, 1]))
