"""The vocabulary of a model input: what each variate is, and why a cell is not an observation."""

from enum import IntEnum


class VariateType(IntEnum):
    """Role of a variate."""

    TARGET = 0
    HISTORICAL = 1
    FUTURE = 2


class MaskType(IntEnum):
    """Reason of values for being masked.

    ``WITHHELD`` marks time steps the model must predict.
    """

    VALID = 0
    PAD = 1
    MISSING = 2
    CENSORED = 3
    WITHHELD = 4


def round_up(value: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is ``>= value``."""
    return -(-value // multiple) * multiple
