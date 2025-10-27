"""
Configuration module for the Legal Agentic RAG system.
Loads environment variables and provides centralized configuration.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration class."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "legal-agentic-rag")
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "singapore-family-court")
    PINECONE_EMBEDDING_MODEL: str = "llama-text-embed-v2"
    PINECONE_EMBEDDING_DIMENSION: int = 1024
    
    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Application Configuration
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Agent Configuration
    MAX_ITERATIONS: int = 5
    VECTOR_SEARCH_TOP_K: int = 5
    
    @classmethod
    def validate(cls) -> None:
        """Validate that all required configuration is present."""
        required_vars = [
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            ("PINECONE_API_KEY", cls.PINECONE_API_KEY),
            ("NEO4J_URI", cls.NEO4J_URI),
            ("NEO4J_PASSWORD", cls.NEO4J_PASSWORD),
        ]
        
        missing = [var_name for var_name, var_value in required_vars if not var_value]
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please check your .env file."
            )


# Validate configuration on import
if Config.ENVIRONMENT != "test":
    try:
        Config.validate()
    except ValueError as e:
        print(f"Warning: {e}")


# Export singleton instance
config = Config()

