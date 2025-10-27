"""
Unit tests for configuration module.
"""

import os
import pytest
from src.config import Config


class TestConfig:
    """Test cases for Config class."""
    
    def test_config_has_required_attributes(self):
        """Test that Config has all required attributes."""
        assert hasattr(Config, 'OPENAI_API_KEY')
        assert hasattr(Config, 'OPENAI_MODEL')
        assert hasattr(Config, 'PINECONE_API_KEY')
        assert hasattr(Config, 'PINECONE_INDEX_NAME')
        assert hasattr(Config, 'NEO4J_URI')
        assert hasattr(Config, 'NEO4J_USERNAME')
        assert hasattr(Config, 'NEO4J_PASSWORD')
    
    def test_openai_model_default(self):
        """Test that OpenAI model defaults to gpt-4o-mini."""
        assert Config.OPENAI_MODEL == "gpt-4o-mini"
    
    def test_pinecone_embedding_model(self):
        """Test that Pinecone embedding model is set correctly."""
        assert Config.PINECONE_EMBEDDING_MODEL == "llama-text-embed-v2"
    
    def test_max_iterations_is_positive(self):
        """Test that MAX_ITERATIONS is a positive integer."""
        assert isinstance(Config.MAX_ITERATIONS, int)
        assert Config.MAX_ITERATIONS > 0
    
    def test_vector_search_top_k_is_positive(self):
        """Test that VECTOR_SEARCH_TOP_K is a positive integer."""
        assert isinstance(Config.VECTOR_SEARCH_TOP_K, int)
        assert Config.VECTOR_SEARCH_TOP_K > 0
    
    def test_validate_method_exists(self):
        """Test that validate method exists."""
        assert hasattr(Config, 'validate')
        assert callable(Config.validate)

