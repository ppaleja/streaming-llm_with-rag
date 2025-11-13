"""Heuristic trigger for deciding when to call the retriever (skeleton).

Provide lightweight heuristics or a pluggable policy to decide when retrieval
should be invoked during streaming generation.
"""
from typing import Any, Dict


class RetrievalTrigger:
    def __init__(self, config: Dict = None):
        self.config = config or {}

    def should_trigger(self, context: Any) -> bool:
        """Return True when retrieval should be triggered based on `context`.

        `context` is opaque here (could be model state, recent tokens, or a query).
        Implementations should inspect it and decide.
        """
        # stub implementation: never trigger
        return False
