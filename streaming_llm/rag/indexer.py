"""Incremental indexer wrapper for evicted segments (skeleton).

This module should wrap a vector store or index (FAISS, LlamaIndex, etc.).
It provides methods to add segments and query the index.
"""
from typing import List, Dict, Optional


class Indexer:
    def __init__(self, index_path: str = None):
        """Initialize indexer. Optionally load existing index from `index_path`.
        """
        self.index_path = index_path
        # internal state / vector index handle

    def add_segment(self, segment: Dict) -> str:
        """Add a segment to the index and return an entry id. (stub)
        """
        raise NotImplementedError()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Return top_k matching segments with scores. (stub)
        """
        raise NotImplementedError()

    def save(self) -> None:
        """Persist index to disk. (stub)
        """
        raise NotImplementedError()

    def load(self) -> None:
        """Load index from disk. (stub)
        """
        raise NotImplementedError()
