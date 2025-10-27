"""
Unit tests for graph search tool.
Tests use actual Neo4j credentials but perform read-only operations.
"""

import pytest
from src.tools.graph_search import GraphSearchTool, get_graph_search_tool
from src.config import config


class TestGraphSearchTool:
    """Test cases for GraphSearchTool."""
    
    @pytest.fixture
    def graph_tool(self):
        """Fixture to get GraphSearchTool instance."""
        return get_graph_search_tool()
    
    def test_tool_initialization(self, graph_tool):
        """Test that GraphSearchTool initializes correctly."""
        assert graph_tool is not None
        assert graph_tool.uri == config.NEO4J_URI
        assert graph_tool.database == config.NEO4J_DATABASE
    
    def test_singleton_pattern(self):
        """Test that get_graph_search_tool returns the same instance."""
        tool1 = get_graph_search_tool()
        tool2 = get_graph_search_tool()
        assert tool1 is tool2
    
    def test_driver_connection(self, graph_tool):
        """Test that Neo4j driver is connected."""
        assert graph_tool.driver is not None
        # Verify connectivity
        graph_tool.driver.verify_connectivity()
    
    def test_execute_cypher_simple_query(self, graph_tool):
        """Test executing a simple Cypher query."""
        # Simple read-only query
        results = graph_tool.execute_cypher("MATCH (n) RETURN count(n) as count LIMIT 1")
        assert isinstance(results, list)
        if results:
            assert 'count' in results[0]
    
    def test_natural_language_search_returns_list(self, graph_tool):
        """Test that natural language search returns a list."""
        results = graph_tool.natural_language_search("Find all cases")
        assert isinstance(results, list)
        # Results may be empty if no data in Neo4j yet
        if results:
            assert all(source.type == "graph" for source in results)
    
    def test_search_method(self, graph_tool):
        """Test the main search method."""
        results = graph_tool.search("What cases involve custody?")
        assert isinstance(results, list)

