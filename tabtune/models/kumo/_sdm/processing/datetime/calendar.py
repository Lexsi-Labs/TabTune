# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from tabtune.models.kumo._sdm import NaT, Stype, TableTensor
from tabtune.models.kumo._sdm.processing import Processor

US_PER_MINUTE = 60 * 1_000_000
US_PER_HOUR = 60 * US_PER_MINUTE
US_PER_DAY = 24 * US_PER_HOUR


class AddCalendarFields(Processor):
    r"""Add numerical calendar fields derived from datetime columns.

    Args:
        fields: The calendar fields to add.
        encoding: How to encode each field. ``"raw"`` uses raw integer values.
            ``"cyclic"`` uses sine and cosine values.
    """

    handles_stypes = frozenset({Stype.datetime})
    requires_fit = False

    def __init__(
        self,
        fields: Sequence[
            Literal[
                "minute",
                "hour",
                "weekday",
                "day_of_month",
                "month",
            ]
        ],
        encoding: Literal["raw", "cyclic"] = "raw",
    ) -> None:
        super().__init__()

        if len(fields) != len(set(fields)):
            raise ValueError("Expected datetime fields to be unique")

        self.fields = fields
        self.encoding = encoding

    def _transform(self, table: TableTensor) -> TableTensor:
        if len(self.fields) == 0:
            return table

        datetime = table.datetime
        na_mask = datetime == NaT

        days: Tensor | None = None
        if len({"weekday", "day_of_month", "month"} & set(self.fields)) > 0:
            days = datetime.div(US_PER_DAY, rounding_mode="floor")

        year = month = day = None
        if len({"day_of_month", "month"} & set(self.fields)) > 0:
            assert days is not None
            year, month, day = _civil_from_days(days)

        time_of_day: Tensor | None = None
        if len({"minute", "hour"} & set(self.fields)) > 0:
            time_of_day = datetime.remainder(US_PER_DAY)

        outs: dict[str, Tensor] = {}
        for field in self.fields:
            if field == "minute":
                assert time_of_day is not None
                out = time_of_day.div(US_PER_MINUTE, rounding_mode="floor")
                outs[field] = out.remainder(60)
                continue

            if field == "hour":
                assert time_of_day is not None
                out = time_of_day.div(US_PER_HOUR, rounding_mode="floor")
                outs[field] = out
                continue

            if field == "weekday":
                assert days is not None
                outs[field] = (days + 3).remainder(7)
                continue

            if field == "month":
                assert month is not None
                outs[field] = month - 1
                continue

            if field == "day_of_month":
                assert day is not None
                outs[field] = day - 1
                continue

            raise ValueError(
                f"{self.__class__.__name__!r} received unsupported "
                f"field {field!r}"
            )

        if self.encoding == "cyclic":
            for field, out in outs.items():
                if field == "minute":
                    outs[field] = out / 60.0
                elif field == "hour":
                    outs[field] = out / 24.0
                elif field == "weekday":
                    outs[field] = out / 7.0
                elif field == "month":
                    outs[field] = out / 12.0
                else:
                    assert field == "day_of_month"
                    assert year is not None
                    assert month is not None
                    period = torch.tensor(
                        [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                        device=year.device,
                    )
                    period = period[month]
                    leap_year = (year.remainder(4) == 0) & (
                        (year.remainder(100) != 0) | (year.remainder(400) == 0)
                    )
                    period = torch.where(leap_year & (month == 2), 29, period)
                    outs[field] = out / period

        out = torch.stack(list(outs.values()), dim=-1)
        out = out.to(table.numerical.dtype)
        out = out.masked_fill(na_mask.unsqueeze(-1), float("nan"))
        out = out.flatten(-2, -1)

        if self.encoding == "cyclic":
            out *= 2 * torch.pi
            out = torch.stack([out.sin(), out.cos()], dim=-1).flatten(-2, -1)

            columns = tuple(
                f"{column}__{field}__{fn}"
                for column in table.columns[Stype.datetime]
                for field in self.fields
                for fn in ("sin", "cos")
            )
        else:
            assert self.encoding == "raw"
            columns = tuple(
                f"{column}__{field}"
                for column in table.columns[Stype.datetime]
                for field in self.fields
            )

        out_table = TableTensor(
            columns={Stype.numerical: columns},
            numerical=out,
        )

        return cast(TableTensor, torch.cat([table, out_table], dim=-1))

    def __repr__(self, *, indent: int = 0) -> str:
        field_repr = "".join(
            f"{' ' * (indent + 4)}{field!r},\n" for field in self.fields
        )
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{' ' * (indent + 2)}fields=[\n"
            f"{field_repr}"
            f"{' ' * (indent + 2)}],\n"
            f"{' ' * (indent + 2)}encoding={self.encoding!r},\n"
            f"{' ' * indent})"
        )


def _civil_from_days(days: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert days since the Unix epoch to Gregorian year, month, and day."""
    shifted = days + 719_468
    era = shifted.div(146_097, rounding_mode="floor")
    day_of_era = shifted - era * 146_097
    year_of_era = (
        day_of_era
        - day_of_era.div(1_460, rounding_mode="floor")
        + day_of_era.div(36_524, rounding_mode="floor")
        - day_of_era.div(146_096, rounding_mode="floor")
    ).div(365, rounding_mode="floor")
    year = year_of_era + era * 400
    day_of_year = day_of_era - (
        365 * year_of_era
        + year_of_era.div(4, rounding_mode="floor")
        - year_of_era.div(100, rounding_mode="floor")
    )
    month_prime = (5 * day_of_year + 2).div(153, rounding_mode="floor")
    month_offset = (153 * month_prime + 2).div(5, rounding_mode="floor")
    day = day_of_year - month_offset + 1
    month = month_prime + torch.where(month_prime < 10, 3, -9)
    year = year + (month <= 2)
    return year, month, day
