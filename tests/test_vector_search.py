"""
Unit tests for vector search tool.
Tests use actual Pinecone credentials but perform read-only operations.
"""

import pytest
from src.tools.vector_search import VectorSearchTool, get_vector_search_tool
from src.config import config


class TestVectorSearchTool:
    """Test cases for VectorSearchTool."""
    
    @pytest.fixture
    def vector_tool(self):
        """Fixture to get VectorSearchTool instance."""
        return get_vector_search_tool()
    
    def test_tool_initialization(self, vector_tool):
        """Test that VectorSearchTool initializes correctly."""
        assert vector_tool is not None
        assert vector_tool.index_name == config.PINECONE_INDEX_NAME
        assert vector_tool.namespace == config.PINECONE_NAMESPACE
    
    def test_singleton_pattern(self):
        """Test that get_vector_search_tool returns the same instance."""
        tool1 = get_vector_search_tool()
        tool2 = get_vector_search_tool()
        assert tool1 is tool2
    
    def test_semantic_search_returns_list(self, vector_tool):
        """Test that semantic search returns a list of sources."""
        results = vector_tool.semantic_search(
            query="custody of children",
            top_k=3
        )
        assert isinstance(results, list)
        # Results may be empty if no data in Pinecone yet
        if results:
            assert all(source.type == "vector" for source in results)
    
    def test_search_with_query(self, vector_tool):
        """Test search method with query parameter."""
        results = vector_tool.search(query="matrimonial assets", top_k=2)
        assert isinstance(results, list)
    
    def test_search_with_no_params_returns_empty(self, vector_tool):
        """Test that search with no parameters returns empty list."""
        results = vector_tool.search()
        assert results == []
    
    def test_fetch_by_ids_returns_list(self, vector_tool):
        """Test that fetch_by_ids returns a list."""
        # Use dummy IDs - will return empty if they don't exist
        results = vector_tool.fetch_by_ids(["nonexistent_id"])
        assert isinstance(results, list)

