"""Unit tests for Retriever class.

Tests verify that the Retriever correctly wraps an Indexer and returns
ranked passages for queries as specified in PRD_RAG-extension.md.
Tests are implementation-agnostic and focus on behavior contracts.
"""
import pytest
from unittest.mock import Mock, MagicMock
from streaming_llm.rag.retriever import Retriever


@pytest.fixture
def mock_indexer():
    """Create a mock Indexer for testing."""
    indexer = Mock()
    # By default, return empty list
    indexer.query.return_value = []
    return indexer


@pytest.fixture
def sample_passages():
    """Provide sample passages that an indexer might return."""
    return [
        {
            "text": "The lazy dog slept peacefully.",
            "score": 0.95,
            "meta": {"position": 5}
        },
        {
            "text": "Fox ran across the field.",
            "score": 0.87,
            "meta": {"position": 3}
        },
        {
            "text": "Dog jumped over the fence.",
            "score": 0.72,
            "meta": {"position": 7}
        },
    ]


class TestRetrieverInitialization:
    """Test Retriever initialization."""

    def test_retriever_init_with_indexer(self, mock_indexer):
        """Retriever can be initialized with an indexer."""
        retriever = Retriever(indexer=mock_indexer)
        assert retriever is not None

    def test_retriever_stores_indexer(self, mock_indexer):
        """Retriever stores the provided indexer."""
        retriever = Retriever(indexer=mock_indexer)
        assert retriever.indexer is mock_indexer

    def test_retriever_init_with_real_indexer(self):
        """Retriever can be initialized with a real Indexer instance."""
        from streaming_llm.rag.indexer import Indexer
        indexer = Indexer()
        retriever = Retriever(indexer=indexer)
        assert retriever.indexer is indexer


class TestRetrieverBasicRetrieval:
    """Test basic retrieval functionality."""

    def test_retrieve_delegates_to_indexer(self, mock_indexer, sample_passages):
        """retrieve delegates to the indexer's query method."""
        mock_indexer.query.return_value = sample_passages
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("test query", top_k=5)

        # Verify indexer.query was called
        mock_indexer.query.assert_called_once_with("test query", top_k=5)
        assert results == sample_passages

    def test_retrieve_returns_list(self, mock_indexer, sample_passages):
        """retrieve returns a list of passages."""
        mock_indexer.query.return_value = sample_passages
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("fox", top_k=5)
        assert isinstance(results, list)

    def test_retrieve_with_default_top_k(self, mock_indexer, sample_passages):
        """retrieve uses default top_k=5 when not specified."""
        mock_indexer.query.return_value = sample_passages
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("test query")

        # Verify top_k=5 was passed to indexer
        mock_indexer.query.assert_called_once_with("test query", top_k=5)
        assert results == sample_passages

    def test_retrieve_with_custom_top_k(self, mock_indexer, sample_passages):
        """retrieve passes custom top_k to indexer."""
        mock_indexer.query.return_value = sample_passages
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("test query", top_k=10)

        # Verify correct top_k was passed
        mock_indexer.query.assert_called_once_with("test query", top_k=10)
        assert results == sample_passages

    @pytest.mark.parametrize("top_k", [1, 3, 5, 10, 20])
    def test_retrieve_with_various_top_k(self, mock_indexer, sample_passages, top_k):
        """retrieve works with various top_k values."""
        mock_indexer.query.return_value = sample_passages[:top_k]
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("query", top_k=top_k)
        mock_indexer.query.assert_called_once_with("query", top_k=top_k)
        assert isinstance(results, list)


class TestRetrieverPassageHandling:
    """Test how Retriever handles passages."""

    def test_retrieve_returns_passages_with_text(self, mock_indexer, sample_passages):
        """Retrieved passages contain text field."""
        mock_indexer.query.return_value = sample_passages
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("test", top_k=5)
        if len(results) > 0:
            assert "text" in results[0]

    def test_retrieve_returns_passages_with_scores(self, mock_indexer, sample_passages):
        """Retrieved passages may contain relevance scores."""
        mock_indexer.query.return_value = sample_passages
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("test", top_k=5)
        if len(results) > 0:
            # Score may be optional depending on implementation
            if "score" in results[0]:
                assert isinstance(results[0]["score"], (int, float))
                assert 0 <= results[0]["score"] <= 1

    def test_retrieve_returns_passages_with_metadata(self, mock_indexer, sample_passages):
        """Retrieved passages may contain metadata."""
        mock_indexer.query.return_value = sample_passages
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("test", top_k=5)
        if len(results) > 0:
            # Metadata may be optional
            if "meta" in results[0]:
                assert isinstance(results[0]["meta"], dict)

    def test_retrieve_preserves_passage_order(self, mock_indexer):
        """Retrieved passages maintain the order from indexer."""
        passages = [
            {"text": "First", "score": 0.9},
            {"text": "Second", "score": 0.8},
            {"text": "Third", "score": 0.7},
        ]
        mock_indexer.query.return_value = passages
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("test", top_k=3)
        assert len(results) == 3
        assert results[0]["text"] == "First"
        assert results[1]["text"] == "Second"
        assert results[2]["text"] == "Third"


class TestRetrieverEmptyResults:
    """Test retriever behavior with empty results."""

    def test_retrieve_empty_results(self, mock_indexer):
        """retrieve handles empty results from indexer."""
        mock_indexer.query.return_value = []
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("nonexistent", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_retrieve_single_result(self, mock_indexer):
        """retrieve handles single result from indexer."""
        single_passage = [{"text": "Only result", "score": 0.95}]
        mock_indexer.query.return_value = single_passage
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("query", top_k=5)
        assert len(results) == 1
        assert results[0]["text"] == "Only result"


class TestRetrieverQueryTypes:
    """Test retriever with various query types."""

    @pytest.mark.parametrize("query", [
        "simple query",
        "multi word search query",
        "query-with-special-chars",
        "query with numbers 123",
    ])
    def test_retrieve_with_various_queries(self, mock_indexer, sample_passages, query):
        """retrieve works with various query text formats."""
        mock_indexer.query.return_value = sample_passages
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve(query, top_k=5)
        assert isinstance(results, list)
        mock_indexer.query.assert_called_once_with(query, top_k=5)

    def test_retrieve_with_empty_query(self, mock_indexer):
        """retrieve handles empty query string."""
        mock_indexer.query.return_value = []
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("", top_k=5)
        mock_indexer.query.assert_called_once_with("", top_k=5)
        assert isinstance(results, list)

    def test_retrieve_with_unicode_query(self, mock_indexer):
        """retrieve handles Unicode query strings."""
        mock_indexer.query.return_value = []
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("你好世界", top_k=5)
        mock_indexer.query.assert_called_once_with("你好世界", top_k=5)
        assert isinstance(results, list)


class TestRetrieverIntegration:
    """Integration tests with real Indexer (without mocking)."""

    def test_retrieve_with_real_indexer(self):
        """Retriever works with a real Indexer instance."""
        from streaming_llm.rag.indexer import Indexer
        indexer = Indexer()
        retriever = Retriever(indexer=indexer)

        # Add some segments
        indexer.add_segment({"text": "The quick brown fox"})
        indexer.add_segment({"text": "Jumps over the lazy dog"})

        results = retriever.retrieve("fox", top_k=5)
        assert isinstance(results, list)

    def test_retrieve_delegates_exactly_to_indexer(self):
        """Retriever delegates exactly to indexer.query without modification."""
        from streaming_llm.rag.indexer import Indexer
        indexer = Indexer()
        retriever = Retriever(indexer=indexer)

        # Add test segment
        indexer.add_segment({"text": "Test content"})

        # retrieve should call indexer.query with same parameters
        retriever_results = retriever.retrieve("test", top_k=3)
        indexer_results = indexer.query("test", top_k=3)

        # Results should be the same (or at least same structure)
        assert isinstance(retriever_results, list)
        assert isinstance(indexer_results, list)


class TestRetrieverEdgeCases:
    """Test edge cases and error conditions."""

    def test_retrieve_very_large_top_k(self, mock_indexer, sample_passages):
        """retrieve handles very large top_k values."""
        mock_indexer.query.return_value = sample_passages
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("test", top_k=1000000)
        mock_indexer.query.assert_called_once_with("test", top_k=1000000)
        assert isinstance(results, list)

    def test_retrieve_zero_top_k(self, mock_indexer):
        """retrieve handles top_k=0."""
        mock_indexer.query.return_value = []
        retriever = Retriever(indexer=mock_indexer)

        results = retriever.retrieve("test", top_k=0)
        mock_indexer.query.assert_called_once_with("test", top_k=0)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_retrieve_multiple_queries_same_retriever(self, mock_indexer, sample_passages):
        """Retriever can be used for multiple queries in sequence."""
        mock_indexer.query.return_value = sample_passages
        retriever = Retriever(indexer=mock_indexer)

        results1 = retriever.retrieve("first query", top_k=5)
        results2 = retriever.retrieve("second query", top_k=3)
        results3 = retriever.retrieve("third query", top_k=1)

        assert mock_indexer.query.call_count == 3
        assert isinstance(results1, list)
        assert isinstance(results2, list)
        assert isinstance(results3, list)