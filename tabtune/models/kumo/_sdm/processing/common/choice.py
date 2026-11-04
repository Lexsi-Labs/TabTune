# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import Literal, cast

import torch
from torch.nn import ModuleList

from tabtune.models.kumo._sdm import EnsembleTable
from tabtune.models.kumo._sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor


class Choice(EnsembleProcessor, EnsembleInvertibleMixin):
    """Route each table or ensemble member through one selected option.

    Options are selected when the processor is fitted. Only selected options
    are fitted, and refitting selects again.

    Args:
        args: Candidate processors or stateless callables.
        method: How to select options. ``"random"`` samples uniformly;
            ``"round_robin"`` assigns options by ensemble member position and
            selects the first option for a single table.
    """

    requires_fit = True

    def __init__(
        self,
        *args: object,
        method: Literal["random", "round_robin"] = "random",
    ) -> None:
        super().__init__()
        options = []
        for arg in args:
            options.append(EnsembleProcessor.as_processor(arg))
        self.options: ModuleList[EnsembleProcessor] = ModuleList(options)
        self.handles_stypes = frozenset(
            stype for option in self.options for stype in option.handles_stypes
        )
        self.method = method
        self._option_ids: tuple[int, ...] = ()

    def get_extra_state(self) -> tuple[int, ...]:
        r""":meta private:"""  # noqa: D415
        return self._option_ids

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._option_ids = cast(tuple[int, ...], state)

    def _draw_option_ids(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[int, ...]:
        if self.method == "round_robin":
            return tuple(
                member_id % len(self.options)
                for member_id in range(len(ensemble_table))
            )

        assert self.method == "random"
        device = (
            next(ensemble_table._iter_groups()).device
            if generator is None
            else generator.device
        )
        return tuple(
            torch.randint(
                len(self.options),
                (len(ensemble_table),),
                generator=generator,
                device=device,
            ).tolist()
        )

    def _tables_by_option(
        self,
        ensemble_table: EnsembleTable,
    ) -> Iterator[tuple[int, EnsembleTable]]:
        member_ids_by_option: dict[int, list[int]] = {}
        for member_id, option_id in enumerate(self._option_ids):
            member_ids_by_option.setdefault(option_id, []).append(member_id)
        # Select lazily, since selecting members may copy them.
        for option_id, member_ids in sorted(member_ids_by_option.items()):
            yield option_id, ensemble_table[member_ids]

    def _gather_outputs(
        self,
        ensemble_table: EnsembleTable,
        outputs: dict[int, EnsembleTable],
    ) -> EnsembleTable:
        tables = []
        member_ids = []
        next_member_id_by_option: dict[int, int] = {}
        for option_id in self._option_ids:
            tables.append(outputs[option_id])
            member_id = next_member_id_by_option.get(option_id, 0)
            member_ids.append(member_id)
            next_member_id_by_option[option_id] = member_id + 1
        return ensemble_table.gather_members(
            tables=tables,
            member_ids=member_ids,
        )

    def _check_num_members(self, ensemble_table: EnsembleTable) -> None:
        if len(self._option_ids) != len(ensemble_table):
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._option_ids)} ensemble members, but got "
                f"{len(ensemble_table)}."
            )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._option_ids = self._draw_option_ids(
            ensemble_table,
            generator=generator,
        )
        for option_id, table in self._tables_by_option(ensemble_table):
            self.options[option_id].fit_ensemble(table, generator=generator)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self._option_ids = self._draw_option_ids(
            ensemble_table,
            generator=generator,
        )
        outputs = {
            option_id: self.options[option_id].fit_transform_ensemble(
                table,
                generator=generator,
            )
            for option_id, table in self._tables_by_option(ensemble_table)
        }
        return self._gather_outputs(ensemble_table, outputs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        self._check_num_members(ensemble_table)
        outputs = {
            option_id: self.options[option_id].transform_ensemble(table)
            for option_id, table in self._tables_by_option(ensemble_table)
        }
        return self._gather_outputs(ensemble_table, outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        self._check_num_members(ensemble_table)
        outputs = {}
        for option_id, table in self._tables_by_option(ensemble_table):
            processor = self.options[option_id]
            if not isinstance(processor, EnsembleInvertibleMixin):
                raise TypeError(
                    f"{processor.__class__.__name__!r} is not invertible."
                )
            outputs[option_id] = processor.inverse_transform_ensemble(table)
        return self._gather_outputs(ensemble_table, outputs)

    def __repr__(self, *, indent: int = 0) -> str:
        inner = ",\n".join(
            option.__repr__(indent=indent + 2) for option in self.options
        )
        if self.method != "random":
            inner += f",\n{' ' * (indent + 2)}method={self.method!r}"
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{inner},\n"
            f"{' ' * indent})"
        )
