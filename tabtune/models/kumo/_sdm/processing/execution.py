# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import NamedTuple, cast

import torch
from torch import Tensor

import tabtune.models.kumo._sdm.processing as sp
from tabtune.models.kumo._sdm import EnsembleTable, Recipe, RelatedTables, Stype, TableTensor
from tabtune.models.kumo._sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor


class MemberContext(NamedTuple):
    """Transformed context tables for one ensemble member."""

    x: TableTensor
    y: TableTensor
    related_tables: RelatedTables[TableTensor] | None
    #: Semantic types of the raw context columns before ``recipe.features``.
    input_stypes: Mapping[str, Stype]


class MemberQuery(NamedTuple):
    """Transformed query tables for one ensemble member."""

    x: TableTensor
    related_tables: RelatedTables[TableTensor] | None


class RecipeExecution:
    """Recipe execution manager during model processing."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe

        self._related_processors: Mapping[str, EnsembleProcessor] | None = None
        self._related_locations: (
            Mapping[str, tuple[tuple[int, int], ...]] | None
        ) = None
        self._num_estimators: int | None = None
        self._y_locations: tuple[tuple[int, int], ...] | None = None

    @property
    def num_members(self) -> int:
        r"""The number of fitted members."""
        assert self._y_locations is not None
        return len(self._y_locations)

    def fit_transform(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        y: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables | None,
        *,
        num_members: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[MemberContext, ...]:
        """Fit and transform context data."""
        # Transform target first to be able to resolve task type. Inverse
        # target transforms require distinct member assignment:
        y = _to_ensemble_table(y, num_members, expand=True)
        y = self.recipe.target.fit_transform_ensemble(y, generator=generator)

        self._num_estimators = num_members
        self._y_locations = y._locations

        task_dispatchers = tuple(
            module
            for processor in (self.recipe.features, self.recipe.output)
            for module in processor.modules()
            if isinstance(module, sp.TaskDispatch)
        )
        if len(task_dispatchers) > 0:
            if any(group.size(-1) != 1 for group in y._iter_groups()):
                raise ValueError(
                    "Expected the transformed target to contain exactly one "
                    "column"
                )

            if all(
                group.numerical.size(-1) == 1 for group in y._iter_groups()
            ):
                task = "regression"
            elif all(
                group.categorical.size(-1) == 1 for group in y._iter_groups()
            ):
                task = "classification"
            else:
                raise ValueError(
                    "'Recipe.target' must resolve to a single task type"
                )

            for task_dispatcher in task_dispatchers:
                task_dispatcher._task = task

        self._related_processors = None
        self._related_locations = None
        related_ensembles: Mapping[str, EnsembleTable] = {}
        if related_tables is not None:
            self._related_processors = {}
            self._related_locations = {}
            for name, table in related_tables.tables.items():
                processor = copy.deepcopy(self.recipe.features)
                for module in processor.modules():
                    if isinstance(module, sp.TableDispatch):
                        module._route = "related"
                self._related_processors[name] = processor
                ensemble_table = _to_ensemble_table(table, num_members)
                self._related_locations[name] = ensemble_table._locations
                related_ensembles[name] = processor.fit_transform_ensemble(
                    ensemble_table,
                    generator=generator,
                )
                if len(related_ensembles[name]) != self.num_members:
                    raise ValueError(
                        "Expected inputs to map to the same number of "
                        "ensemble members"
                    )

        for module in self.recipe.features.modules():
            if isinstance(module, sp.TableDispatch):
                module._route = "task"

        inputs = _to_ensemble_table(x, num_members)
        x = self.recipe.features.fit_transform_ensemble(
            inputs,
            generator=generator,
        )
        if len(x) != self.num_members:
            raise ValueError(
                "Expected inputs to map to the same number of ensemble members"
            )

        members: list[MemberContext] = []
        for member_id in range(self.num_members):
            related_tables_i: RelatedTables[TableTensor] | None = None
            if related_tables is not None:
                related_tables_i = RelatedTables(
                    tables={
                        name: table[member_id]
                        for name, table in related_ensembles.items()
                    },
                    relationships=related_tables.relationships,
                    task_links=related_tables.task_links,
                )
            members.append(
                MemberContext(
                    x=x[member_id],
                    y=y[member_id],
                    related_tables=related_tables_i,
                    input_stypes=inputs[member_id].stypes,
                )
            )

        return tuple(members)

    def transform(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables | None,
    ) -> tuple[MemberQuery, ...]:
        """Transform query data."""
        x = _to_ensemble_table(x, self._num_estimators)
        x = self.recipe.features.transform_ensemble(x)
        if len(x) != self.num_members:
            raise ValueError(
                "Expected inputs to map to the same number of ensemble members"
            )

        related_ensembles: Mapping[str, EnsembleTable] = {}
        if related_tables is not None:
            assert self._related_processors is not None
            assert self._related_locations is not None
            for name, table in related_tables.tables.items():
                processor = self._related_processors[name]
                ensemble_table = _to_ensemble_table(
                    table, self._num_estimators
                )
                ensemble_table = _align_to_fitted_groups(
                    ensemble_table,
                    self._related_locations[name],
                )
                related_ensembles[name] = processor.transform_ensemble(
                    ensemble_table
                )
                if len(related_ensembles[name]) != self.num_members:
                    raise ValueError(
                        "Expected inputs to map to the same number of "
                        "ensemble members"
                    )

        members: list[MemberQuery] = []
        for member_id in range(self.num_members):
            related_tables_i: RelatedTables[TableTensor] | None = None
            if related_tables is not None:
                related_tables_i = RelatedTables(
                    tables={
                        name: table[member_id]
                        for name, table in related_ensembles.items()
                    },
                    relationships=related_tables.relationships,
                    task_links=related_tables.task_links,
                )
            members.append(
                MemberQuery(
                    x=x[member_id],
                    related_tables=related_tables_i,
                )
            )

        return tuple(members)

    def inverse_transform_target(
        self,
        outputs: Sequence[TableTensor],
    ) -> tuple[TableTensor, ...]:
        """Invert fitted target transforms on member outputs."""
        assert len(outputs) == self.num_members

        # Reconstruct the group layout of the transformed target:
        assert self._y_locations is not None
        num_groups = max(group for group, _ in self._y_locations) + 1
        groups: list[list[TableTensor | None]] = [
            [] for _ in range(num_groups)
        ]
        for group_id, _ in self._y_locations:
            groups[group_id].append(None)
        for i, (group_id, position) in enumerate(self._y_locations):
            groups[group_id][position] = outputs[i]

        table = EnsembleTable(
            groups=[
                cast(
                    TableTensor,
                    group[0].unsqueeze(0)  # type: ignore
                    if len(group) == 1
                    else torch.stack(group, dim=0),  # type: ignore
                )
                for group in groups
            ],
            locations=self._y_locations,
        )

        if not isinstance(self.recipe.target, EnsembleInvertibleMixin):
            raise RuntimeError("Target recipe is not invertible")
        table = self.recipe.target.inverse_transform_ensemble(table)
        return tuple(table[i] for i in range(len(table)))

    def transform_output(
        self,
        outputs: Sequence[TableTensor],
    ) -> TableTensor:
        """Apply ``recipe.output`` to member outputs."""
        if len(outputs) == 1:
            out = outputs[0].unsqueeze(0)
        else:
            expected = set(outputs[0].columns[Stype.numerical])
            for output in outputs[1:]:
                if set(output.columns[Stype.numerical]) != expected:
                    raise ValueError(
                        "Expected all model outputs to have the same columns "
                        "before applying 'Recipe.output'. Ensure every target "
                        "contains the same set of classes."
                    )

            out = torch.stack(list(outputs), dim=0)

        return self.recipe.output.transform(cast(TableTensor, out))


def _align_to_fitted_groups(
    ensemble_table: EnsembleTable,
    fitted_locations: Sequence[tuple[int, int]],
) -> EnsembleTable:
    """Align query groups with the fitted related-table groups."""
    fitted_locations = tuple(fitted_locations)
    if ensemble_table._locations == fitted_locations:
        return ensemble_table

    members_by_fitted_position: list[dict[int, list[int]]] = [
        {} for _ in range(max(group for group, _ in fitted_locations) + 1)
    ]
    for member_id, (group, position) in enumerate(fitted_locations):
        members_by_fitted_position[group].setdefault(position, []).append(
            member_id
        )

    groups = []
    locations = list(fitted_locations)
    for group_id, fitted_positions in enumerate(members_by_fitted_position):
        positions_and_members = tuple(sorted(fitted_positions.items()))
        sources_by_position = tuple(
            tuple(
                dict.fromkeys(
                    ensemble_table._locations[member_id]
                    for member_id in member_ids
                )
            )
            for _, member_ids in positions_and_members
        )

        # A fitted position may fan out only when its state can broadcast over
        # the entire group; otherwise states could map to the wrong queries.
        if len(sources_by_position) > 1 and any(
            len(sources) > 1 for sources in sources_by_position
        ):
            raise ValueError(
                "Cannot align query tables with fitted related-table groups"
            )

        if len(sources_by_position) == 1:
            source_locations = sources_by_position[0]
            position_by_source = {
                source: position
                for position, source in enumerate(source_locations)
            }
            for member_id in positions_and_members[0][1]:
                locations[member_id] = (
                    group_id,
                    position_by_source[ensemble_table._locations[member_id]],
                )
        else:
            source_locations = tuple(
                sources[0] for sources in sources_by_position
            )

        tables = tuple(
            ensemble_table._groups[group][position]
            for group, position in source_locations
        )
        # Reuse one shared query without materializing a copy per fitted state.
        if all(source == source_locations[0] for source in source_locations):
            table = tables[0]
            output = cast(
                TableTensor,
                table.unsqueeze(0).expand(len(tables), *table.size()),
            )
        else:
            packed_groups, _ = EnsembleTable._pack_tables(tables)
            if len(packed_groups) != 1:
                raise ValueError(
                    "Cannot align query tables with fitted related-table "
                    "groups"
                )
            output = packed_groups[0]
        groups.append(output)

    return EnsembleTable(groups=groups, locations=locations)


def _to_ensemble_table(
    x: Tensor | TableTensor | EnsembleTable,
    num_estimators: int | None,
    *,
    expand: bool = False,
) -> EnsembleTable:
    r"""Convert model input data into an ensemble-aware representation.

    Args:
        x: The input table.
        num_estimators: Number of estimators to represent. If ``None``, a 2D
            input creates one estimator and higher-rank inputs infer the
            estimator count from their leading dimension.
        expand: Whether to expand shared inputs into distinct logical member
            positions.
    """
    if isinstance(x, EnsembleTable):
        if num_estimators is not None and num_estimators != len(x):
            raise ValueError(
                f"Expected {num_estimators} members in 'EnsembleTable' "
                f"(got {len(x)})"
            )
        if expand:
            groups = [x.expanded_group(i) for i in range(x.num_groups)]
            locations = []
            next_pos = [0] * x.num_groups
            for i, _ in x._locations:
                locations.append((i, next_pos[i]))
                next_pos[i] += 1
            x = EnsembleTable(
                groups=tuple(groups),
                locations=tuple(locations),
            )
        if len(x) < 1:
            raise ValueError("'num_estimators' needs to be positive")
        return x

    if not isinstance(x, TableTensor):
        x = TableTensor.from_tensor(x)

    # Treat leading dimension as ensemble dimension:
    if x.dim() > 2 and num_estimators is None:
        locations = tuple((0, i) for i in range(x.size(0)))
        x = EnsembleTable(groups=(x,), locations=locations)
        if len(x) < 1:
            raise ValueError("'num_estimators' needs to be positive")
        return x

    num_estimators = 1 if num_estimators is None else num_estimators
    if num_estimators < 1:
        raise ValueError("'num_estimators' needs to be positive")

    # Add a leading ensemble dimension:
    x = x.unsqueeze(0)
    if expand:
        x = x.expand(num_estimators, *x.size()[1:])

    if x.size(0) == 1:
        locations = ((0, 0),) * num_estimators
    else:
        locations = tuple((0, i) for i in range(num_estimators))

    return EnsembleTable(groups=(cast(TableTensor, x),), locations=locations)
