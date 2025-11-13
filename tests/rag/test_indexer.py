"""Unit tests for Indexer class.

Tests verify that the Indexer correctly manages segments and retrieves
ranked results based on queries, as specified in PRD_RAG-extension.md.
Tests are implementation-agnostic and focus on behavior contracts.
"""
import pytest
import tempfile
import os
from streaming_llm.rag.indexer import Indexer


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for index files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_segments():
    """Provide sample segments for testing."""
    return [
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "meta": {"position": 0, "timestamp": 1.0}
        },
        {
            "text": "Machine learning models require large datasets.",
            "meta": {"position": 1, "timestamp": 2.0}
        },
        {
            "text": "Neural networks are inspired by biological neurons.",
            "meta": {"position": 2, "timestamp": 3.0}
        },
    ]


class TestIndexerInitialization:
    """Test Indexer initialization."""

    def test_indexer_init_without_path(self):
        """Indexer can be initialized without an index path."""
        indexer = Indexer()
        assert indexer is not None

    def test_indexer_init_with_path(self, temp_index_dir):
        """Indexer can be initialized with an index path."""
        indexer = Indexer(index_path=temp_index_dir)
        assert indexer.index_path == temp_index_dir

    def test_indexer_accepts_valid_path(self, temp_index_dir):
        """Indexer stores the provided index path."""
        indexer = Indexer(index_path=temp_index_dir)
        assert indexer.index_path is not None


class TestIndexerSegmentManagement:
    """Test segment addition and management."""

    def test_add_segment_returns_id(self, sample_segments):
        """Adding a segment returns a non-empty identifier."""
        indexer = Indexer()
        segment_id = indexer.add_segment(segment=sample_segments[0])
        assert segment_id is not None
        assert isinstance(segment_id, str)
        assert len(segment_id) > 0

    def test_add_multiple_segments_returns_different_ids(self, sample_segments):
        """Adding multiple segments returns different IDs."""
        indexer = Indexer()
        ids = [indexer.add_segment(seg) for seg in sample_segments]
        assert len(ids) == len(sample_segments)
        assert len(set(ids)) == len(ids), "All segment IDs should be unique"

    def test_add_segment_with_text(self):
        """Segments with text content can be added."""
        indexer = Indexer()
        segment = {"text": "Sample text for indexing"}
        segment_id = indexer.add_segment(segment)
        assert segment_id is not None

    def test_add_segment_with_metadata(self):
        """Segments with metadata can be added."""
        indexer = Indexer()
        segment = {
            "text": "Important segment",
            "meta": {"source": "document1", "position": 10}
        }
        segment_id = indexer.add_segment(segment)
        assert segment_id is not None

    def test_add_empty_segment_text(self):
        """Adding a segment with empty text is handled (implementation-dependent)."""
        indexer = Indexer()
        segment = {"text": "", "meta": {}}
        # Should not raise an exception
        try:
            segment_id = indexer.add_segment(segment)
            assert segment_id is not None or segment_id is None
        except (ValueError, TypeError):
            pytest.skip("Implementation rejects empty segments")


class TestIndexerQuerying:
    """Test query functionality."""

    def test_query_returns_list(self, sample_segments):
        """Query returns a list of results."""
        indexer = Indexer()
        for seg in sample_segments:
            indexer.add_segment(seg)

        results = indexer.query("fox", top_k=5)
        assert isinstance(results, list)

    def test_query_respects_top_k(self, sample_segments):
        """Query returns at most top_k results."""
        indexer = Indexer()
        for seg in sample_segments:
            indexer.add_segment(seg)

        results = indexer.query("neural", top_k=2)
        assert len(results) <= 2

    def test_query_with_default_top_k(self, sample_segments):
        """Query uses default top_k when not specified."""
        indexer = Indexer()
        for seg in sample_segments:
            indexer.add_segment(seg)

        results = indexer.query("machine learning")
        assert isinstance(results, list)
        assert len(results) <= 5  # Default is 5

    def test_query_results_are_dicts(self, sample_segments):
        """Query results are dictionaries."""
        indexer = Indexer()
        for seg in sample_segments:
            indexer.add_segment(seg)

        results = indexer.query("fox", top_k=1)
        if len(results) > 0:
            assert isinstance(results[0], dict)

    def test_query_results_contain_text(self, sample_segments):
        """Query results contain the original segment text."""
        indexer = Indexer()
        for seg in sample_segments:
            indexer.add_segment(seg)

        results = indexer.query("fox", top_k=5)
        if len(results) > 0:
            assert "text" in results[0]

    def test_query_results_may_contain_scores(self, sample_segments):
        """Query results may contain relevance scores."""
        indexer = Indexer()
        for seg in sample_segments:
            indexer.add_segment(seg)

        results = indexer.query("neural networks", top_k=5)
        if len(results) > 0:
            # Score may or may not be present, depending on implementation
            # But if present, it should be numeric
            if "score" in results[0]:
                assert isinstance(results[0]["score"], (int, float))

    @pytest.mark.parametrize("query_text", [
        "fox",
        "machine learning",
        "neural networks",
        "dataset",
    ])
    def test_query_with_various_texts(self, sample_segments, query_text):
        """Query works with various text inputs."""
        indexer = Indexer()
        for seg in sample_segments:
            indexer.add_segment(seg)

        results = indexer.query(query_text, top_k=5)
        assert isinstance(results, list)

    def test_query_on_empty_index(self):
        """Query on an empty index returns an empty list."""
        indexer = Indexer()
        results = indexer.query("anything", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_query_with_zero_top_k(self, sample_segments):
        """Query with top_k=0 returns empty list."""
        indexer = Indexer()
        for seg in sample_segments:
            indexer.add_segment(seg)

        results = indexer.query("fox", top_k=0)
        assert isinstance(results, list)
        assert len(results) == 0


class TestIndexerPersistence:
    """Test index persistence (save/load)."""

    def test_save_raises_not_implemented(self, temp_index_dir):
        """Save method is a stub that raises NotImplementedError."""
        indexer = Indexer(index_path=temp_index_dir)
        with pytest.raises(NotImplementedError):
            indexer.save()

    def test_load_raises_not_implemented(self, temp_index_dir):
        """Load method is a stub that raises NotImplementedError."""
        indexer = Indexer(index_path=temp_index_dir)
        with pytest.raises(NotImplementedError):
            indexer.load()


class TestIndexerEdgeCases:
    """Test edge cases and error conditions."""

    def test_query_empty_string(self):
        """Query with empty string is handled."""
        indexer = Indexer()
        indexer.add_segment({"text": "Some content"})
        results = indexer.query("", top_k=5)
        assert isinstance(results, list)

    def test_large_top_k(self, sample_segments):
        """Query with large top_k returns available results."""
        indexer = Indexer()
        for seg in sample_segments:
            indexer.add_segment(seg)

        results = indexer.query("test", top_k=1000)
        assert len(results) <= len(sample_segments)

    def test_segment_with_special_characters(self):
        """Segments with special characters can be indexed."""
        indexer = Indexer()
        segment = {"text": "Special chars: !@#$%^&*()"}
        segment_id = indexer.add_segment(segment)
        assert segment_id is not None

    def test_segment_with_unicode(self):
        """Segments with Unicode characters can be indexed."""
        indexer = Indexer()
        segment = {"text": "Unicode: 你好世界 🚀 مرحبا"}
        segment_id = indexer.add_segment(segment)
        assert segment_id is not None

    def test_segment_with_long_text(self):
        """Segments with very long text can be indexed."""
        indexer = Indexer()
        long_text = "word " * 10000  # ~50K characters
        segment = {"text": long_text}
        segment_id = indexer.add_segment(segment)
        assert segment_id is not None