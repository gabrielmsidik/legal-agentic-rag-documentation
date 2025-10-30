"""
Tests for the reranking module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.tools.reranker import RerankerTool, get_reranker
from src.models.schemas import Source


class TestRerankerTool:
    """Test cases for RerankerTool."""
    
    @pytest.fixture
    def reranker(self):
        """Create a reranker instance for testing."""
        with patch('src.tools.reranker.Pinecone'):
            return RerankerTool()
    
    @pytest.fixture
    def sample_sources(self):
        """Create sample sources for testing."""
        return [
            Source(
                type="vector",
                chunk_id="chunk_1",
                data="This is about financial hardship in matrimonial cases",
                metadata={"case_extract_name": "case_extract_2024_1", "score": 0.85}
            ),
            Source(
                type="vector",
                chunk_id="chunk_2",
                data="Discussion of asset division in family law",
                metadata={"case_extract_name": "case_extract_2024_2", "score": 0.75}
            ),
            Source(
                type="graph",
                chunk_id=None,
                data={"case_number": "2024_SGHCF_1", "summary": "Case about maintenance", "outcome": "Accepted"},
                metadata={"similarity_score": 0.80}
            ),
            Source(
                type="graph",
                chunk_id=None,
                data={"name": "Women's Charter", "section": "Section 59", "summary": "Maintenance provisions"},
                metadata={"similarity_score": 0.70}
            ),
            Source(
                type="vector",
                chunk_id="chunk_3",
                data="Joint accounts and matrimonial property",
                metadata={"case_extract_name": "case_extract_2024_3", "score": 0.65}
            ),
        ]
    
    def test_extract_text_for_vector_source(self, reranker):
        """Test text extraction from vector sources."""
        source = Source(
            type="vector",
            chunk_id="chunk_1",
            data="Sample text content",
            metadata={}
        )
        text = reranker._extract_text_for_reranking(source)
        assert text == "Sample text content"
    
    def test_extract_text_for_graph_source(self, reranker):
        """Test text extraction from graph sources."""
        source = Source(
            type="graph",
            chunk_id=None,
            data={
                "case_number": "2024_SGHCF_1",
                "name": "ABC v XYZ",
                "summary": "Case summary",
                "outcome": "Accepted"
            },
            metadata={}
        )
        text = reranker._extract_text_for_reranking(source)
        assert "Case: 2024_SGHCF_1" in text
        assert "Name: ABC v XYZ" in text
        assert "Summary: Case summary" in text
        assert "Outcome: Accepted" in text
    
    def test_rerank_sources_empty_list(self, reranker):
        """Test reranking with empty source list."""
        result = reranker.rerank_sources("test query", [], top_n=5)
        assert result == []
    
    def test_rerank_sources_fewer_than_top_n(self, reranker, sample_sources):
        """Test reranking when sources are fewer than top_n."""
        sources = sample_sources[:2]
        result = reranker.rerank_sources("test query", sources, top_n=5)
        assert len(result) == 2
        assert result == sources
    
    def test_rerank_sources_returns_top_n(self, reranker, sample_sources):
        """Test that reranking returns top_n sources."""
        # When reranking fails (no real API), it should return first top_n sources
        result = reranker.rerank_sources("financial hardship", sample_sources, top_n=5)

        # Verify results - should return all 5 sources (fallback behavior)
        assert len(result) == 5
        # Verify we got the first 5 sources
        assert result[0].chunk_id == "chunk_1"
        assert result[1].chunk_id == "chunk_2"
    
    @patch('src.tools.reranker.Pinecone')
    def test_rerank_sources_api_failure_fallback(self, mock_pinecone_class, sample_sources):
        """Test fallback behavior when reranking API fails."""
        # Setup mock to raise exception
        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc
        mock_pc.inference.rerank.side_effect = Exception("API Error")
        
        # Create reranker with mocked Pinecone
        reranker = RerankerTool()
        
        # Perform reranking - should fallback gracefully
        result = reranker.rerank_sources("test query", sample_sources, top_n=5)
        
        # Should return first 5 sources in original order
        assert len(result) == 5
        assert result == sample_sources[:5]
    
    def test_rerank_sources_mixed_types(self, reranker, sample_sources):
        """Test that reranking handles mixed vector and graph sources."""
        # Verify we have both types
        vector_count = sum(1 for s in sample_sources if s.type == "vector")
        graph_count = sum(1 for s in sample_sources if s.type == "graph")
        
        assert vector_count > 0
        assert graph_count > 0
        
        # Extract text for all sources
        texts = [reranker._extract_text_for_reranking(s) for s in sample_sources]
        
        # Verify all texts are non-empty
        assert all(len(t) > 0 for t in texts)


class TestRerankerIntegration:
    """Integration tests for reranker with agent."""

    @pytest.fixture
    def reranker_fixture(self):
        """Create a reranker instance for integration tests."""
        with patch('src.tools.reranker.Pinecone'):
            return RerankerTool()

    @patch('src.tools.reranker.Pinecone')
    def test_reranker_singleton(self, mock_pinecone_class):
        """Test that get_reranker returns singleton instance."""
        mock_pinecone_class.return_value = MagicMock()

        reranker1 = get_reranker()
        reranker2 = get_reranker()

        assert reranker1 is reranker2

    def test_reranker_preserves_source_data(self, reranker_fixture):
        """Test that reranking preserves all source data."""
        sources = [
            Source(
                type="vector",
                chunk_id="chunk_1",
                data="Test data 1",
                metadata={"case_extract_name": "case_1", "score": 0.8}
            ),
            Source(
                type="vector",
                chunk_id="chunk_2",
                data="Test data 2",
                metadata={"case_extract_name": "case_2", "score": 0.7}
            ),
        ]

        result = reranker_fixture.rerank_sources("test", sources, top_n=2)

        # Verify original data is preserved (fallback returns first top_n)
        assert result[0].chunk_id == "chunk_1"
        assert result[0].data == "Test data 1"
        assert result[0].metadata is not None
        assert result[0].metadata.get("case_extract_name") == "case_1"
        # Metadata should be preserved even if reranking fails
        assert result[0].metadata.get("score") == 0.8

