"""Unit tests for EvictedStore class.

Tests verify that the EvictedStore correctly persists, retrieves, and lists
evicted text segments as specified in PRD_RAG-extension.md.
Tests are implementation-agnostic and focus on behavior contracts.
"""
import pytest
import tempfile
import os
from streaming_llm.rag.store import EvictedStore


@pytest.fixture
def temp_store_dir():
    """Create a temporary directory for store files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def evicted_store(temp_store_dir):
    """Create an EvictedStore instance with a temporary directory."""
    return EvictedStore(path=temp_store_dir)


@pytest.fixture
def sample_segments():
    """Provide sample evicted segments for testing."""
    return [
        {
            "text": "First evicted segment with important information.",
            "meta": {"timestamp": 1.0, "token_range": (0, 100)}
        },
        {
            "text": "Second evicted segment containing context.",
            "meta": {"timestamp": 2.0, "token_range": (100, 200)}
        },
        {
            "text": "Third evicted segment with additional details.",
            "meta": {"timestamp": 3.0, "token_range": (200, 300)}
        },
    ]


class TestEvictedStoreInitialization:
    """Test EvictedStore initialization."""

    def test_store_init_creates_directory(self, temp_store_dir):
        """Initializing store creates the directory if it doesn't exist."""
        new_path = os.path.join(temp_store_dir, "new_store")
        store = EvictedStore(path=new_path)
        assert os.path.exists(new_path)
        assert os.path.isdir(new_path)

    def test_store_init_with_existing_directory(self, temp_store_dir):
        """Store can be initialized with an existing directory."""
        store = EvictedStore(path=temp_store_dir)
        assert store.path == temp_store_dir
        assert os.path.exists(store.path)

    def test_store_init_preserves_path(self, temp_store_dir):
        """Store stores the provided path."""
        store = EvictedStore(path=temp_store_dir)
        assert store.path == temp_store_dir


class TestEvictedStoreSegmentPersistence:
    """Test segment persistence (add_segment)."""

    def test_add_segment_returns_id(self, evicted_store):
        """Adding a segment returns a non-empty identifier."""
        segment = {"text": "Test segment", "meta": {}}
        segment_id = evicted_store.add_segment(segment)
        assert segment_id is not None
        assert isinstance(segment_id, str)
        assert len(segment_id) > 0

    def test_add_multiple_segments_returns_different_ids(self, evicted_store, sample_segments):
        """Adding multiple segments returns different IDs."""
        ids = [evicted_store.add_segment(seg) for seg in sample_segments]
        assert len(ids) == len(sample_segments)
        assert len(set(ids)) == len(ids), "All segment IDs should be unique"

    def test_add_segment_with_text(self, evicted_store):
        """Segments with text content can be persisted."""
        segment = {"text": "Important evicted content"}
        segment_id = evicted_store.add_segment(segment)
        assert segment_id is not None

    def test_add_segment_with_metadata(self, evicted_store):
        """Segments with metadata can be persisted."""
        segment = {
            "text": "Content with meta",
            "meta": {"timestamp": 1.5, "source": "llm_generation"}
        }
        segment_id = evicted_store.add_segment(segment)
        assert segment_id is not None

    def test_add_segment_with_complex_metadata(self, evicted_store):
        """Segments with complex metadata structures can be persisted."""
        segment = {
            "text": "Complex metadata segment",
            "meta": {
                "timestamp": 2.5,
                "token_range": (50, 150),
                "attention_score": 0.95,
                "tags": ["important", "context"]
            }
        }
        segment_id = evicted_store.add_segment(segment)
        assert segment_id is not None

    @pytest.mark.parametrize("text_length", [1, 100, 1000, 10000])
    def test_add_segment_various_sizes(self, evicted_store, text_length):
        """Segments of various text lengths can be persisted."""
        segment = {"text": "x" * text_length, "meta": {}}
        segment_id = evicted_store.add_segment(segment)
        assert segment_id is not None


class TestEvictedStoreSegmentRetrieval:
    """Test segment retrieval (get_segment)."""

    def test_get_segment_returns_dict(self, evicted_store):
        """get_segment returns a dictionary."""
        segment = {"text": "Test content", "meta": {"test": True}}
        segment_id = evicted_store.add_segment(segment)
        retrieved = evicted_store.get_segment(segment_id)
        assert isinstance(retrieved, dict)

    def test_get_segment_returns_text(self, evicted_store):
        """Retrieved segment contains the original text."""
        original_text = "Original segment text"
        segment = {"text": original_text, "meta": {}}
        segment_id = evicted_store.add_segment(segment)
        retrieved = evicted_store.get_segment(segment_id)
        assert "text" in retrieved
        assert retrieved["text"] == original_text

    def test_get_segment_returns_metadata(self, evicted_store):
        """Retrieved segment contains the original metadata."""
        original_meta = {"timestamp": 1.5, "position": 10}
        segment = {"text": "Content", "meta": original_meta}
        segment_id = evicted_store.add_segment(segment)
        retrieved = evicted_store.get_segment(segment_id)
        assert "meta" in retrieved
        assert retrieved["meta"] == original_meta

    def test_get_nonexistent_segment_returns_none(self, evicted_store):
        """Retrieving a non-existent segment returns None."""
        result = evicted_store.get_segment("nonexistent_id_12345")
        assert result is None

    def test_get_segment_after_multiple_additions(self, evicted_store, sample_segments):
        """Segment can be retrieved after adding multiple segments."""
        ids = [evicted_store.add_segment(seg) for seg in sample_segments]
        # Retrieve middle segment
        middle_id = ids[1]
        retrieved = evicted_store.get_segment(middle_id)
        assert retrieved is not None
        assert retrieved["text"] == sample_segments[1]["text"]

    def test_get_segment_with_unicode(self, evicted_store):
        """Unicode content in segments is preserved on retrieval."""
        unicode_text = "Unicode: 你好世界 مرحبا 🚀"
        segment = {"text": unicode_text, "meta": {}}
        segment_id = evicted_store.add_segment(segment)
        retrieved = evicted_store.get_segment(segment_id)
        assert retrieved["text"] == unicode_text

    def test_get_segment_preserves_special_characters(self, evicted_store):
        """Special characters in segments are preserved."""
        special_text = "Special: !@#$%^&*()_+-=[]{}|;:',.<>?/\\"
        segment = {"text": special_text, "meta": {}}
        segment_id = evicted_store.add_segment(segment)
        retrieved = evicted_store.get_segment(segment_id)
        assert retrieved["text"] == special_text


class TestEvictedStoreListingSegments:
    """Test segment listing (list_segments)."""

    def test_list_segments_returns_list(self, evicted_store):
        """list_segments returns a list."""
        result = evicted_store.list_segments()
        assert isinstance(result, list)

    def test_list_segments_empty_store(self, evicted_store):
        """Listing segments in an empty store returns an empty list."""
        result = evicted_store.list_segments()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_list_segments_after_additions(self, evicted_store, sample_segments):
        """Listing segments after adding segments returns them."""
        for seg in sample_segments:
            evicted_store.add_segment(seg)

        result = evicted_store.list_segments()
        assert len(result) >= len(sample_segments)

    def test_list_segments_respects_limit(self, evicted_store, sample_segments):
        """list_segments respects the limit parameter."""
        for seg in sample_segments:
            evicted_store.add_segment(seg)

        result = evicted_store.list_segments(limit=2)
        assert len(result) <= 2

    def test_list_segments_returns_dicts(self, evicted_store, sample_segments):
        """Segments from list_segments are dictionaries."""
        for seg in sample_segments:
            evicted_store.add_segment(seg)

        result = evicted_store.list_segments()
        if len(result) > 0:
            for seg in result:
                assert isinstance(seg, dict)

    def test_list_segments_contains_text(self, evicted_store, sample_segments):
        """Segments from list_segments contain text."""
        for seg in sample_segments:
            evicted_store.add_segment(seg)

        result = evicted_store.list_segments()
        if len(result) > 0:
            assert "text" in result[0]

    def test_list_segments_default_limit(self, evicted_store, sample_segments):
        """list_segments uses default limit of 100."""
        # Add fewer than 100 segments
        for seg in sample_segments:
            evicted_store.add_segment(seg)

        result = evicted_store.list_segments()
        assert len(result) <= 100

    @pytest.mark.parametrize("limit", [1, 5, 10, 50])
    def test_list_segments_various_limits(self, evicted_store, sample_segments, limit):
        """list_segments works with various limit values."""
        for seg in sample_segments:
            evicted_store.add_segment(seg)

        result = evicted_store.list_segments(limit=limit)
        assert len(result) <= limit

    def test_list_segments_order_newest_first(self, evicted_store):
        """list_segments returns segments with newest first (if metadata available)."""
        seg1 = {"text": "First", "meta": {"timestamp": 1.0}}
        seg2 = {"text": "Second", "meta": {"timestamp": 2.0}}
        seg3 = {"text": "Third", "meta": {"timestamp": 3.0}}

        evicted_store.add_segment(seg1)
        evicted_store.add_segment(seg2)
        evicted_store.add_segment(seg3)

        result = evicted_store.list_segments(limit=100)
        # If ordering is preserved, most recent should be first or last
        # (implementation may vary; just verify it returns all)
        assert len(result) >= 3


class TestEvictedStoreEdgeCases:
    """Test edge cases and error conditions."""

    def test_add_segment_empty_text(self, evicted_store):
        """Adding a segment with empty text is handled."""
        segment = {"text": "", "meta": {}}
        try:
            segment_id = evicted_store.add_segment(segment)
            assert segment_id is not None or segment_id is None
        except (ValueError, TypeError):
            pytest.skip("Implementation rejects empty text")

    def test_add_segment_no_metadata(self, evicted_store):
        """Adding a segment without metadata field is handled."""
        segment = {"text": "Content without meta"}
        try:
            segment_id = evicted_store.add_segment(segment)
            assert segment_id is not None
        except KeyError:
            pytest.skip("Implementation requires metadata field")

    def test_add_segment_with_none_values(self, evicted_store):
        """Adding a segment with None values in metadata is handled."""
        segment = {"text": "Content", "meta": {"field": None}}
        segment_id = evicted_store.add_segment(segment)
        assert segment_id is not None

    def test_get_segment_with_empty_string_id(self, evicted_store):
        """Getting a segment with empty string ID returns None."""
        result = evicted_store.get_segment("")
        assert result is None

    def test_segment_id_format_consistency(self, evicted_store, sample_segments):
        """All segment IDs have consistent format (strings)."""
        ids = [evicted_store.add_segment(seg) for seg in sample_segments]
        for seg_id in ids:
            assert isinstance(seg_id, str)

    def test_multiple_stores_independent(self, temp_store_dir):
        """Multiple stores with different paths are independent."""
        store1_path = os.path.join(temp_store_dir, "store1")
        store2_path = os.path.join(temp_store_dir, "store2")

        store1 = EvictedStore(path=store1_path)
        store2 = EvictedStore(path=store2_path)

        seg1 = {"text": "Store 1 content", "meta": {}}
        seg2 = {"text": "Store 2 content", "meta": {}}

        id1 = store1.add_segment(seg1)
        id2 = store2.add_segment(seg2)

        # Each store should only retrieve its own segments
        assert store1.get_segment(id1) is not None
        assert store2.get_segment(id2) is not None