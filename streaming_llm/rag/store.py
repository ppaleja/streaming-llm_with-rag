"""Evicted segments persistent store (skeleton).

This module provides a simple API to persist and retrieve evicted text segments.
Implementations can back this by files, sqlite, or an object store.
"""
from typing import List, Dict, Optional
import os


class EvictedStore:
    """Simple append-only store for evicted text segments.

    Each segment is a dict containing at least:
      - "id": unique id
      - "text": raw text of the evicted segment
      - "meta": optional metadata (timestamps, token ranges)
    """

    def __init__(self, path: str):
        """Create a store rooted at `path`.
        """
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def add_segment(self, segment: Dict) -> str:
        """Persist a new segment and return its id. (stub)
        """
        # TODO: implement persistence
        raise NotImplementedError()

    def get_segment(self, segment_id: str) -> Optional[Dict]:
        """Load a segment by id. (stub)
        """
        raise NotImplementedError()

    def list_segments(self, limit: int = 100) -> List[Dict]:
        """List latest segments (stub).
        """
        raise NotImplementedError()
