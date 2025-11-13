"""High-level retriever abstraction (skeleton).

This module should provide a simple `Retriever` class that wraps an `Indexer` and
returns ranked passages for a query.
"""
from typing import List, Dict


class Retriever:
    def __init__(self, indexer):
        """Create a retriever over the given `indexer` (stub).
        """
        self.indexer = indexer

    def retrieve(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Return top_k candidate passages for `query_text` with scores. (stub)
        """
        return self.indexer.query(query_text, top_k=top_k)
