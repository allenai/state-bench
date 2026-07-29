"""Canonical, formatter-independent state-tracking data model."""

from __future__ import annotations

import hashlib
import json
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
    """An evaluated semantic trace, with no surface-form decisions."""

    instance_id: str
    complexity: Complexity
    num_variables: int
    num_values: int
    operations: tuple[Operation, ...]
    query: tuple[int, ...]
    final_state: tuple[int, ...]

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
    ) -> StateTrackingInstance:
        semantic_record = {
            "complexity": complexity.value,
            "num_variables": num_variables,
            "num_values": num_values,
            "operations": [operation.to_record() for operation in operations],
            "query": list(query),
            "final_state": list(final_state),
        }
        digest = hashlib.sha256(
            json.dumps(semantic_record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        return cls(
            instance_id=digest,
            complexity=complexity,
            num_variables=num_variables,
            num_values=num_values,
            operations=operations,
            query=query,
            final_state=final_state,
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
        }


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
    }
