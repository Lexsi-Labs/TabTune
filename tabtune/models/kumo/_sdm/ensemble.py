# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import copy
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Generic, TypeVar, cast, overload
from typing_extensions import Self

import torch
from torch import Tensor

from tabtune.models.kumo._sdm import Stype, StypeLike
from tabtune.models.kumo._sdm.tensor import TableTensor
from tabtune.models.kumo._sdm.tensor.mixin import DeviceMixin

T = TypeVar("T")


class EnsembleData(abc.ABC, Generic[T]):
    """Represent an ordered collection of ensemble member values.

    Args:
        groups: Values containing the ensemble members.
        locations: Group index and position for each ensemble member.
    """

    _groups: tuple[T, ...]
    _locations: tuple[tuple[int, int], ...]

    def __init__(
        self,
        groups: Sequence[T],
        locations: Sequence[tuple[int, int]],
    ) -> None:
        self._groups = tuple(groups)
        self._locations = tuple(locations)

    def __len__(self) -> int:
        return len(self._locations)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice | Sequence[int]) -> Self: ...

    def __getitem__(self, index: int | slice | Sequence[int]) -> T | Self:
        if isinstance(index, int):
            group_id, position = self._locations[index]
            return self._select_member(self._groups[group_id], position)
        if isinstance(index, slice):
            index = range(*index.indices(len(self)))
        return self._select_members(index)

    def _iter_groups(self) -> Iterator[T]:
        return iter(self._groups)

    @staticmethod
    @abc.abstractmethod
    def _select_member(group: T, position: int) -> T:
        pass

    @abc.abstractmethod
    def _select_members(self, member_ids: Sequence[int]) -> Self:
        pass


class EnsembleTable(DeviceMixin, EnsembleData[TableTensor]):
    """Store and group input :class:`~sdm.tensor.TableTensor` as an ensemble.

    Each ensemble member is associated with one
    :class:`~sdm.tensor.TableTensor`.
    Shared tables are stored only once. Compatible tables are stacked along a
    leading dimension and form a group.

    Args:
        groups: Sequence of :class:`~sdm.tensor.TableTensor`, optionally
            stacked along their leading dimension.
        locations: ``(group, batch)`` location of each ensemble member.
    """

    @classmethod
    def from_table(
        cls,
        table: TableTensor,
        *,
        num_members: int,
    ) -> Self:
        """Create an ensemble, sharing one table across all members.

        Args:
            table: Table used across members.
            num_members: Number of ensemble members.

        Returns:
            An :class:`~sdm.EnsembleTable` with one group.
        """
        return cls(
            groups=(cast(TableTensor, table.unsqueeze(0)),),
            locations=((0, 0),) * num_members,
        )

    @classmethod
    def from_tables(
        cls,
        tables: Sequence[TableTensor],
        member_table_ids: Sequence[int],
    ) -> Self:
        """Create an ensemble table from tables and their member assignments.

        Each referenced input table forms a separate group.
        ``member_table_ids`` contains one table index per member. For
        example, ``(0, 1, 0)`` assigns the first table to members 0 and 2 and
        the second table to member 1.

        Args:
            tables: Tables available to the ensemble members.
            member_table_ids: Index into ``tables`` for each ensemble member.

        Returns:
            An ensemble table preserving member order.
        """
        referenced_table_ids = set(member_table_ids)
        table_ids = tuple(
            table_id
            for table_id in range(len(tables))
            if table_id in referenced_table_ids
        )
        group_ids = {
            table_id: group_id for group_id, table_id in enumerate(table_ids)
        }
        return cls(
            groups=tuple(
                cast(TableTensor, tables[table_id].unsqueeze(0))
                for table_id in table_ids
            ),
            locations=tuple(
                (group_ids[table_id], 0) for table_id in member_table_ids
            ),
        )

    @staticmethod
    def _pack_tables(
        tables: Sequence[TableTensor],
    ) -> tuple[tuple[TableTensor, ...], tuple[tuple[int, int], ...]]:
        compatible_groups: dict[tuple[object, ...], list[int]] = {}
        for index, table in enumerate(tables):
            # Shape, schema, block layout, device, and categorical vocabularies
            # must match for torch.stack to preserve member semantics.
            compatibility_key = (
                tuple(
                    (stype, columns)
                    for stype, columns in table.columns.items()
                ),
                tuple(
                    (
                        stype,
                        type(block),
                        block.size(),
                        block.layout,
                        block.dtype,
                    )
                    for stype, block in table.items()
                ),
                table.device,
                tuple(
                    id(category) for category in table.categorical.categories
                ),
            )
            compatible_groups.setdefault(compatibility_key, []).append(index)

        groups: list[TableTensor] = []
        locations = [(-1, -1)] * len(tables)
        for indices in compatible_groups.values():
            group_index = len(groups)
            groups.append(
                cast(TableTensor, tables[indices[0]].unsqueeze(0))
                if len(indices) == 1
                else cast(
                    TableTensor,
                    torch.stack(
                        tensors=[tables[index] for index in indices],
                        dim=0,
                    ),
                )
            )
            for position, index in enumerate(indices):
                locations[index] = (group_index, position)

        return tuple(groups), tuple(locations)

    def _select_members(self, member_ids: Sequence[int]) -> Self:
        member_ids = tuple(member_ids)
        if member_ids == tuple(range(len(self))):
            return self

        locations = tuple(
            self._locations[member_id] for member_id in member_ids
        )
        positions_by_group: dict[int, dict[int, int]] = {}
        for group_id, position in locations:
            positions = positions_by_group.setdefault(group_id, {})
            positions.setdefault(position, len(positions))

        groups: list[TableTensor] = []
        group_ids: dict[int, int] = {}
        for new_group_id, (group_id, positions) in enumerate(
            positions_by_group.items()
        ):
            group = self._groups[group_id]
            selected_positions = tuple(positions)
            group_ids[group_id] = new_group_id
            start = selected_positions[0]
            step = (
                selected_positions[1] - start
                if len(selected_positions) > 1
                else 1
            )
            stop = start + step * len(selected_positions)
            if selected_positions == tuple(range(group.size(0))):
                groups.append(group)
            elif step > 0 and selected_positions == tuple(
                range(start, stop, step)
            ):
                # Evenly spaced members are selected as a view, not a copy.
                groups.append(group[start:stop:step])
            else:
                groups.append(
                    cast(
                        TableTensor,
                        torch.stack(
                            [
                                group[position]
                                for position in selected_positions
                            ],
                            dim=0,
                        ),
                    )
                )

        ensemble = copy.copy(self)
        ensemble._groups = tuple(groups)
        ensemble._locations = tuple(
            (group_ids[group_id], positions_by_group[group_id][position])
            for group_id, position in locations
        )
        return ensemble

    def gather_members(
        self,
        tables: Sequence[Self],
        member_ids: Sequence[int],
    ) -> Self:
        """Gather members from multiple ensemble tables.

        ``tables[i][member_ids[i]]`` supplies output member ``i``.

        Args:
            tables: Source ensemble table for each output member.
            member_ids: Logical source member position for each output member.

        Returns:
            An ensemble table preserving current group boundaries.
        """
        if len(tables) != len(member_ids):
            raise ValueError("Expected one source member per ensemble table")

        outputs: list[TableTensor] = []
        output_id_by_source: dict[tuple[int, tuple[int, int]], int] = {}
        member_table_ids = []
        for table, member_id in zip(tables, member_ids, strict=True):
            location = table._locations[member_id]
            key = (id(table), location)
            output_id = output_id_by_source.get(key)
            if output_id is None:
                output_id = len(outputs)
                output_id_by_source[key] = output_id
                outputs.append(table[member_id])
            member_table_ids.append(output_id)

        return self.replace_tables(
            tables=outputs,
            member_table_ids=member_table_ids,
        )

    @property
    def num_groups(self) -> int:
        """Return the number of table groups."""
        return len(self._groups)

    def num_members_in_group(self, group_id: int) -> int:
        """Return the number of members of a group.

        Args:
            group_id: Zero-based group index.
        """
        return sum(i == group_id for i, _ in self._locations)

    @staticmethod
    def _select_member(
        group: TableTensor,
        position: int,
    ) -> TableTensor:
        return group[position]

    def expanded_group(self, group_id: int) -> TableTensor:
        """Return the logical members assigned to one group.

        Args:
            group_id: Zero-based group index.
        """
        group = self._groups[group_id]
        positions = tuple(
            position for i, position in self._locations if i == group_id
        )
        if positions == tuple(range(group.size(0))):
            return group
        if group.size(0) == 1:
            return cast(
                TableTensor,
                group.expand(len(positions), *group.size()[1:]),
            )
        index = torch.tensor(positions, device=group.device)
        return cast(TableTensor, group.index_select(0, index))

    def _tensors(self) -> Iterator[Tensor]:
        yield from self._groups

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.replace_groups(
            [cast(TableTensor, fn(group)) for group in self._groups]
        )

    def select_stypes(
        self,
        stypes: StypeLike | Iterable[StypeLike],
    ) -> Self:
        r"""Return an ensemble table containing only ``stypes`` columns.

        Args:
            stypes: The semantic type or semantic types to select.

        Returns:
            An ensemble table preserving its logical member assignment.
        """
        if isinstance(stypes, (str, Stype)):
            stypes = (stypes,)
        stypes = tuple(Stype(stype) for stype in stypes)

        if len(stypes) == 0:
            return self.replace_groups(
                [group.select_columns(()) for group in self._groups]
            )
        return self.replace_groups(
            [group.select_stypes(stypes) for group in self._groups]
        )

    @classmethod
    def concatenate_columns(cls, tables: Sequence[Self]) -> Self:
        r"""Concatenate ensemble tables column-wise by logical member.

        Args:
            tables: Ensemble tables with the same number of logical members.

        Returns:
            An ensemble table preserving logical member order.
        """
        if len(tables) == 0:
            raise ValueError("Expected at least one ensemble table")

        first = tables[0]
        if any(len(table) != len(first) for table in tables[1:]):
            raise ValueError(
                "Cannot concatenate ensemble tables with different member "
                "counts"
            )

        nonempty_tables = tuple(
            table
            for table in tables
            if any(group.size(-1) > 0 for group in table._groups)
        )
        if len(nonempty_tables) > 0:
            tables = nonempty_tables
            first = tables[0]
        if len(tables) == 1:
            return first

        if all(table._locations == first._locations for table in tables[1:]):
            return first.replace_groups(
                [
                    cast(TableTensor, torch.cat(groups, dim=-1))
                    for groups in zip(
                        *(table._groups for table in tables),
                        strict=True,
                    )
                ]
            )

        # TODO: Concatenate compatible groups directly and unpack logical
        # members only when their layouts differ.
        outputs: list[TableTensor] = []
        output_id_by_locations: dict[tuple[tuple[int, int], ...], int] = {}
        member_table_ids = []
        for member_id in range(len(first)):
            locations = tuple(table._locations[member_id] for table in tables)
            output_id = output_id_by_locations.get(locations)
            if output_id is None:
                output_id = len(outputs)
                output_id_by_locations[locations] = output_id
                outputs.append(
                    cast(
                        TableTensor,
                        torch.cat(
                            tuple(table[member_id] for table in tables),
                            dim=-1,
                        ),
                    )
                )
            member_table_ids.append(output_id)

        return cls.from_tables(
            tables=outputs,
            member_table_ids=member_table_ids,
        )

    def replace_tables(
        self,
        tables: Sequence[TableTensor],
        member_table_ids: Sequence[int],
    ) -> Self:
        """Replace member tables.

        Args:
            tables: Replacement tables available to the ensemble members.
            member_table_ids: Index into ``tables`` for each ensemble member.

        Returns:
            An ensemble table preserving existing group boundaries.
        """
        if len(member_table_ids) != len(self):
            raise ValueError("Expected one replacement table per member")

        member_ids_by_group: list[list[int]] = [
            [] for _ in range(self.num_groups)
        ]
        for member_id, (group_id, _) in enumerate(self._locations):
            member_ids_by_group[group_id].append(member_id)

        groups: list[TableTensor] = []
        locations = [(-1, -1)] * len(self)
        for member_ids in member_ids_by_group:
            table_ids = tuple(
                dict.fromkeys(
                    member_table_ids[member_id] for member_id in member_ids
                )
            )
            packed_groups, packed_locations = self._pack_tables(
                tuple(tables[table_id] for table_id in table_ids)
            )
            group_offset = len(groups)
            groups.extend(packed_groups)
            locations_by_table_id = {
                table_id: (group_id + group_offset, position)
                for table_id, (group_id, position) in zip(
                    table_ids, packed_locations, strict=True
                )
            }
            for member_id in member_ids:
                locations[member_id] = locations_by_table_id[
                    member_table_ids[member_id]
                ]

        return self.__class__(groups=groups, locations=locations)

    def replace_groups(self, groups: Sequence[TableTensor]) -> Self:
        """Return an ensemble table with its groups replaced.

        Args:
            groups: One replacement group per current group.

        Returns:
            An ensemble table over ``groups`` with the current member
            assignment.
        """
        if len(groups) != self.num_groups:
            raise ValueError(
                f"Expected one replacement per group "
                f"({self.num_groups}), got {len(groups)}."
            )
        ensemble = copy.copy(self)
        ensemble._groups = tuple(groups)
        return ensemble

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"num_members={len(self)}, num_groups={self.num_groups})"
        )
