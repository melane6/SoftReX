# python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import uuid
from mutpy import operators


@dataclass
class Mutation:
    """Record describing a mutation attempt or applied mutation."""
    id: str
    kind: str
    target: Any
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    original: Any = None
    applied: bool = False

# python
from abc import ABC, abstractmethod
class BaseMutator(ABC):
    """Abstract mutator API. Concrete mutators should override methods."""

    def __init__(self, representation: Any):
        self.representation = representation
        self._mutations: Dict[str, Mutation] = {}

    def _new_id(self) -> str:
        return str(uuid.uuid4())

    def register(self, mutation: Mutation) -> None:
        self._mutations[mutation.id] = mutation

    def list_mutations(self) -> List[Mutation]:
        return list(self._mutations.values())

    @abstractmethod
    def mutate_range(self, start: int, end: int, payload: Any) -> Mutation:
        """Mutate a linear range (lines / statement indices)."""
        op = operators.MutationOperator.mutate(self.representation, start, end, payload)
        raise NotImplementedError

    @abstractmethod
    def mutate_node(self, node_id: Any, payload: Any) -> Mutation:
        """Mutate a specific node (line id, AST node, CFG node)."""
        raise NotImplementedError

    @abstractmethod
    def mutate_block(self, block_id: Any, payload: Any) -> Mutation:
        """Mutate a specific block (AST block, CFG block)."""
        raise NotImplementedError

    @abstractmethod
    def apply_mutation(self, mutation: Mutation) -> None:
        """Apply a previously created mutation to the representation."""
        raise NotImplementedError

    @abstractmethod
    def revert_mutation(self, mutation: Mutation) -> None:
        """Revert a previously applied mutation using stored `original` data."""
        raise NotImplementedError
