# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
from typing_extensions import Self

from torch import Tensor

from tabtune.models.kumo._sdm import EnsembleTable, TableTensor
from tabtune.models.kumo._sdm.relational import RelationalData, Relationship
from tabtune.models.kumo._sdm.relational.join import LEFT_ROW_ID, RIGHT_ROW_ID
from tabtune.models.kumo._sdm.tensor.mixin import DeviceMixin
from tabtune.models.kumo._sdm.tensor.table import TableSchema

if TYPE_CHECKING:
    import graphviz


T = TypeVar("T", bound=TableTensor | EnsembleTable)

TASK_TABLE = "__task_table__"


@dataclass(frozen=True, repr=False)
class TaskLink:
    r"""Link between task rows to a table in relational data.

    Args:
        task_columns: Column names in the task table.
        table: Name of the table referenced by task rows.
        table_columns: Column names in ``table``.
    """

    task_columns: tuple[str, ...]
    table: str
    table_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.task_columns) != len(self.table_columns):
            raise ValueError(
                f"Expected 'task_columns' and 'table_columns' to have the "
                f"same length (got {len(self.task_columns)} and "
                f"{len(self.table_columns)})"
            )

        if len(self.task_columns) == 0:
            raise ValueError(
                "Expected 'task_columns' and 'table_columns' to be non-empty"
            )

        for column in (*self.task_columns, *self.table_columns):
            for reserved in (LEFT_ROW_ID, RIGHT_ROW_ID):
                if column == reserved:
                    raise ValueError(
                        f"Column name {column!r} is reserved for internal "
                        f"row indexing"
                    )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str | Sequence[str]]) -> Self:
        r"""Create a :class:`TaskLink` from a mapping.

        Args:
            mapping: The mapping with ``"task_columns"``, ``"table"``, and
                ``"table_columns"`` entries.
        """
        if "task_column" in mapping:
            task_columns = mapping["task_column"]
        else:
            task_columns = mapping["task_columns"]
        if isinstance(task_columns, str):
            task_columns = (task_columns,)

        table = mapping["table"]
        assert isinstance(table, str)

        if "table_column" in mapping:
            table_columns = mapping["table_column"]
        else:
            table_columns = mapping["table_columns"]
        if isinstance(table_columns, str):
            table_columns = (table_columns,)

        return cls(
            task_columns=tuple(task_columns),
            table=table,
            table_columns=tuple(table_columns),
        )

    def _task_columns_repr(self) -> str:
        if len(self.task_columns) == 1:
            return self.task_columns[0]
        return f"[{','.join(self.task_columns)}]"

    def _table_columns_repr(self) -> str:
        if len(self.table_columns) == 1:
            return f"{self.table}.{self.table_columns[0]}"
        return f"{self.table}.[{','.join(self.table_columns)}]"

    def __repr__(self) -> str:
        return f"{self._task_columns_repr()} > {self._table_columns_repr()}"


@dataclass(frozen=True)
class RelatedTablesSchema:
    r"""The schema of :class:`RelatedTables`.

    Args:
        tables: Table schema keyed by table name.
        relationships: Join relationships among ``tables``.
        task_links: Links from task columns to related ``tables``.
    """

    tables: Mapping[str, TableSchema]
    relationships: tuple[Relationship, ...]
    task_links: tuple[TaskLink, ...]

    def is_subset_of(self, other: RelatedTablesSchema) -> bool:
        r"""Whether this schema is an induced subset of ``other``."""
        for table_name, schema in self.tables.items():
            if schema != other.tables.get(table_name):
                return False

        other_relationships = {
            relationship
            for relationship in other.relationships
            if relationship.left_table in self.tables
            and relationship.right_table in self.tables
        }
        if set(self.relationships) != other_relationships:
            return False

        other_task_links = {
            task_link
            for task_link in other.task_links
            if task_link.table in self.tables
        }
        return set(self.task_links) == other_task_links

    def __hash__(self) -> int:
        return hash(
            (
                frozenset(self.tables.items()),
                self.relationships,
                self.task_links,
            )
        )


@dataclass(frozen=True, init=False, repr=False)
class RelatedTables(DeviceMixin, Generic[T]):
    r"""Task-specific related tables attached to model inputs.

    :class:`RelatedTables` store the relational context provided to a model
    for a particular task table.
    It may contain a sampled subset of a larger :class:`RelationalData`.
    The ``task_links`` describe how rows in the model input match to rows in
    the related tables.

    .. testcode::

        from tabtune.models.kumo._sdm import RelatedTables, TableTensor

        data = RelatedTables(
            tables={
                "users": TableTensor.from_columns(
                    {"user_id": [0, 1]},
                    stypes={"user_id": "id"},
                ),
                "orders": TableTensor.from_columns(
                    {
                        "user_id": [0, 1],
                        "item_id": [10, 11],
                    },
                    stypes={
                        "user_id": "id",
                        "item_id": "id",
                    },
                ),
                "items": TableTensor.from_columns(
                    {"item_id": [10, 11]},
                    stypes={"item_id": "id"},
                ),
            },
            relationships=[
                # Foreign key from orders to users:
                dict(left_table="orders", left_column="user_id", right_table="users", right_column="user_id"),
                # Foreign key from orders to items:
                dict(left_table="orders", left_column="item_id", right_table="items", right_column="item_id"),
            ],
            task_links=[
                # Foreign key in the task table to users:
                dict(task_column="ENTITY", table="users", table_column="user_id")
            ],
        )

    Args:
        tables: Related tables keyed by table name.
        relationships: Join relationships among ``tables``.
        task_links: Links from task columns to related ``tables``.
    """  # noqa: E501

    tables: Mapping[str, T]
    relationships: tuple[Relationship, ...]
    task_links: tuple[TaskLink, ...]

    def __init__(
        self,
        tables: Mapping[str, T],
        relationships: Collection[
            Relationship | Mapping[str, str | Sequence[str]]
        ],
        task_links: Collection[TaskLink | Mapping[str, str | Sequence[str]]],
    ) -> None:

        relationships = tuple(
            relationship
            if isinstance(relationship, Relationship)
            else Relationship.from_mapping(relationship)
            for relationship in relationships
        )

        task_links = tuple(
            task_link
            if isinstance(task_link, TaskLink)
            else TaskLink.from_mapping(task_link)
            for task_link in task_links
        )

        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "relationships", relationships)
        object.__setattr__(self, "task_links", task_links)

    def _tensors(self) -> Iterator[Tensor]:
        for table in self.tables.values():
            if isinstance(table, DeviceMixin):
                yield from table._tensors()
                continue
            yield table

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(
            tables={
                name: table._apply_tensor(fn)
                if isinstance(table, DeviceMixin)
                else cast(TableTensor, fn(table))
                for name, table in self.tables.items()
            },
            relationships=self.relationships,
            task_links=self.task_links,
        )

    @property
    def schema(self) -> RelatedTablesSchema:
        r"""The schema of this related context."""
        table_schemas = {}
        for name, table in self.tables.items():
            if isinstance(table, EnsembleTable):
                schemas = {group.schema for group in table._iter_groups()}
                if len(schemas) != 1:
                    raise ValueError(
                        "'schema' requires each 'EnsembleTable' to have a "
                        "unique table schema"
                    )
                table_schemas[name] = next(iter(schemas))
                continue
            table_schemas[name] = table.schema

        return RelatedTablesSchema(
            tables=table_schemas,
            relationships=self.relationships,
            task_links=self.task_links,
        )

    def is_same_schema(self, other: RelatedTables) -> bool:
        r"""Whether ``other`` has the same schema layout.

        Args:
            other: The object to compare against.
        """
        return self.schema == other.schema

    def select_tables(self, tables: Iterable[str]) -> Self:
        r"""Return related tables containing only ``tables``.

        Args:
            tables: The table names to select.
        """
        tables = set(tables)

        return self.__class__(
            tables={
                table_name: table
                for table_name, table in self.tables.items()
                if table_name in tables
            },
            relationships=tuple(
                relationship
                for relationship in self.relationships
                if relationship.left_table in tables
                and relationship.right_table in tables
            ),
            task_links=tuple(
                task_link
                for task_link in self.task_links
                if task_link.table in tables
            ),
        )

    def replace_tables(self, tables: Mapping[str, T]) -> Self:
        r"""Return related tables with replaced table data.

        Args:
            tables: Related tables keyed by table name.
        """
        if tables.keys() != self.tables.keys():
            raise ValueError("Expected 'tables' to match existing table names")

        return self.__class__(
            tables=tables,
            relationships=self.relationships,
            task_links=self.task_links,
        )

    def to_graphviz(
        self,
        *,
        hide_columns: bool = False,
        **kwargs: Any,
    ) -> graphviz.Graph:
        r"""Return a task visualization of the relational schema.

        Args:
            hide_columns: Whether to hide column name descriptions.
            **kwargs: Additional keyword arguments passed to
                :class:`graphviz.Graph`.
        """
        tables = {}
        for name, table in self.tables.items():
            if isinstance(table, EnsembleTable):
                if len({group.schema for group in table._iter_groups()}) != 1:
                    raise ValueError(
                        "'to_graphviz' requires each 'EnsembleTable' to have "
                        "a unique table schema"
                    )
                tables[name] = table[0]
                continue
            tables[name] = table

        graph = RelationalData(
            tables=tables,
            relationships=self.relationships,
        ).to_graphviz(hide_columns=hide_columns, **kwargs)

        graph.node(TASK_TABLE, label="", shape="point")

        for link in self.task_links:
            label = "\\n".join(
                f" {task_column} > {table_column} "
                for task_column, table_column in zip(
                    link.task_columns, link.table_columns
                )
            )
            graph.edge(
                TASK_TABLE,
                link.table,
                label=label,
                fontsize="11pt",
            )

        return graph

    def __repr__(self) -> str:
        out = f"{self.__class__.__name__}(\n"
        if len(self.tables) > 0:
            out += "  tables={\n"
            out += "".join(
                f"    {key}: {value.__repr__(indent=4)[4:]},\n"  # type: ignore
                for key, value in self.tables.items()
            )
            out += "  },\n"
        else:
            out += "  tables={},\n"
        if len(self.relationships) > 0:
            out += "  relationships=[\n"
            out += "".join(f"    {rel},\n" for rel in self.relationships)
            out += "  ],\n"
        else:
            out += "  relationships=[],\n"
        if len(self.task_links) > 0:
            out += "  task_links=[\n"
            out += "".join(f"    {link},\n" for link in self.task_links)
            out += "  ],\n"
        else:
            out += "  task_links=[],\n"
        out += ")"
        return out

    def _repr_html_(self) -> str:
        import pandas as pd

        rows = []
        for name, table in self.tables.items():
            row: list[Any] = [name]
            if isinstance(table, EnsembleTable):
                num_rows = {group.size(-2) for group in table._iter_groups()}
                row.append(
                    next(iter(num_rows))
                    if len(num_rows) == 1
                    else f"{min(num_rows)} - {max(num_rows)}"
                )
                num_cols = {group.size(-1) for group in table._iter_groups()}
                row.append(
                    next(iter(num_cols))
                    if len(num_cols) == 1
                    else f"{min(num_cols)} - {max(num_cols)}"
                )
                stypes = {
                    stype
                    for g in table._iter_groups()
                    for stype in g.active_stypes
                }
                row.append(", ".join(stypes))
            else:
                row += [
                    table.size(-2),
                    table.size(-1),
                    ", ".join(table.active_stypes),
                ]
            rows.append(row)

        df = pd.DataFrame(
            rows,
            columns=pd.Index(["Table", "Rows", "Columns", "Stypes"]),
        )

        ul = "".join(
            f"<li>"
            f"<code>{escape(link._task_columns_repr())}</code>"
            f" ➡️ "
            f"<code>{escape(link._table_columns_repr())}</code>"
            f"</li>"
            for link in self.task_links
        )
        ul += "".join(
            f"<li>"
            f"<code>{escape(rel._left_columns_repr())}</code>"
            f" ↔️ "
            f"<code>{escape(rel._right_columns_repr())}</code>"
            f"</li>"
            for rel in self.relationships
        )

        return df.to_html(index=False, escape=True) + f"<ul>{ul}</ul>"
