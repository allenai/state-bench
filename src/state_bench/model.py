"""Canonical, formatter-independent state-tracking data model."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Complexity(StrEnum):
    """Algebraic family used to generate a state-tracking instance."""

    R_TRIVIAL = "r-trivial"
    APERIODIC = "aperiodic"
    PERIODIC = "periodic"


class QueryMode(StrEnum):
    ALL = "all"
    ONE = "one"
    SUBSET = "subset"


@dataclass(frozen=True, slots=True)
class AssignConstant:
    target: int
    value: int

    def to_record(self) -> dict[str, int | str]:
        return {"op": "constant", "target": self.target, "value": self.value}


@dataclass(frozen=True, slots=True)
class AssignVariable:
    target: int
    source: int

    def to_record(self) -> dict[str, int | str]:
        return {"op": "variable", "target": self.target, "source": self.source}


type Operation = AssignConstant | AssignVariable


@dataclass(frozen=True, slots=True)
class StateTrackingInstance:
    """An evaluated semantic trace, with no surface-form decisions.

    ``num_variables`` counts ordinary task variables. Periodic source traces
    may additionally carry one scratch register at ``temporary_variable``;
    that register participates in operations and queries but is not part of
    the configured variable-count axis.
    """

    instance_id: str
    complexity: Complexity
    num_variables: int
    num_values: int
    operations: tuple[Operation, ...]
    query: tuple[int, ...]
    final_state: tuple[int, ...]
    temporary_variable: int | None

    @property
    def num_registers(self) -> int:
        return self.num_variables + (self.temporary_variable is not None)

    @property
    def num_initial_assignments(self) -> int:
        return self.num_registers

    @classmethod
    def create(
        cls,
        *,
        complexity: Complexity,
        num_variables: int,
        num_values: int,
        operations: tuple[Operation, ...],
        query: tuple[int, ...],
        final_state: tuple[int, ...],
        temporary_variable: int | None = None,
    ) -> StateTrackingInstance:
        num_registers = num_variables + (temporary_variable is not None)
        if temporary_variable is not None and temporary_variable != num_variables:
            raise ValueError("the temporary variable must immediately follow ordinary variables")
        if len(final_state) != num_registers:
            raise ValueError("final_state length must equal the number of registers")
        digest = _semantic_digest(
            complexity=complexity,
            num_variables=num_variables,
            num_values=num_values,
            operations=operations,
            query=query,
            final_state=final_state,
            temporary_variable=temporary_variable,
        )
        return cls(
            instance_id=digest,
            complexity=complexity,
            num_variables=num_variables,
            num_values=num_values,
            operations=operations,
            query=query,
            final_state=final_state,
            temporary_variable=temporary_variable,
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> StateTrackingInstance:
        semantic = semantic_record_from_dict(record)
        return cls(instance_id=str(record["instance_id"]), **semantic)

    def to_record(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "complexity": self.complexity.value,
            "num_variables": self.num_variables,
            "num_values": self.num_values,
            "num_assignments": len(self.operations),
            "operations": [operation.to_record() for operation in self.operations],
            "query": list(self.query),
            "final_state": list(self.final_state),
            "temporary_variable": self.temporary_variable,
        }


def _semantic_digest(
    *,
    complexity: Complexity,
    num_variables: int,
    num_values: int,
    operations: tuple[Operation, ...],
    query: tuple[int, ...],
    final_state: tuple[int, ...],
    temporary_variable: int | None,
) -> str:
    """Content hash of a trace's semantics, stable across processes and runs.

    The layout is length-prefixed and fixed-width so distinct traces cannot
    collide by field boundary ambiguity: scalar header, then the query and
    final-state arrays each preceded by their length, then the operation count
    followed by a ``(kind, target, argument)`` int triple per operation.
    """

    buffer = bytearray(complexity.value.encode())
    buffer += struct.pack(
        "<iiii",
        num_variables,
        num_values,
        -1 if temporary_variable is None else temporary_variable,
        len(query),
    )
    buffer += struct.pack(f"<{len(query)}i", *query)
    buffer += struct.pack(f"<{len(final_state) + 1}i", len(final_state), *final_state)
    operation_ints: list[int] = []
    for operation in operations:
        if isinstance(operation, AssignConstant):
            operation_ints += (0, operation.target, operation.value)
        else:
            operation_ints += (1, operation.target, operation.source)
    buffer += struct.pack("<i", len(operations))
    buffer += struct.pack(f"<{len(operation_ints)}i", *operation_ints)
    return hashlib.sha256(buffer).hexdigest()[:20]


def semantic_record_from_dict(record: dict[str, Any]) -> dict[str, Any]:
    operations: list[Operation] = []
    for operation in record["operations"]:
        if operation["op"] == "constant":
            operations.append(AssignConstant(target=int(operation["target"]), value=int(operation["value"])))
        elif operation["op"] == "variable":
            operations.append(
                AssignVariable(target=int(operation["target"]), source=int(operation["source"]))
            )
        else:
            raise ValueError(f"unknown operation type: {operation['op']!r}")
    return {
        "complexity": Complexity(record["complexity"]),
        "num_variables": int(record["num_variables"]),
        "num_values": int(record["num_values"]),
        "operations": tuple(operations),
        "query": tuple(int(value) for value in record["query"]),
        "final_state": tuple(int(value) for value in record["final_state"]),
        "temporary_variable": (
            int(record["temporary_variable"])
            if record.get("temporary_variable") is not None
            else None
        ),
    }
