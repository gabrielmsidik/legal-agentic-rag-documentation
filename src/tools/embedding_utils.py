"""
Utility functions for generating and managing embeddings for semantic search.
Integrates with Pinecone's inference API for server-side embedding generation.
"""

import logging
from typing import List, Optional
from pinecone import Pinecone
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for semantic search using Pinecone's inference API.
    Embeddings are stored in Neo4j nodes for hybrid vector-graph search.
    """
    
    def __init__(self):
        """Initialize Pinecone client for embedding generation."""
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.embedding_model = config.PINECONE_EMBEDDING_MODEL
        self.embedding_dimension = config.PINECONE_EMBEDDING_DIMENSION
        
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text using Pinecone's inference API.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return None
            
        try:
            # Use Pinecone's inference API for server-side embedding
            result = self.pc.inference.embed(
                model=self.embedding_model,
                inputs=[text],
                parameters={"input_type": "passage"}
            )
            
            if result and result.data and len(result.data) > 0:
                embedding = result.data[0]["values"]
                logger.debug(f"Generated embedding of dimension {len(embedding)}")
                return embedding
            else:
                logger.error("No embedding returned from Pinecone inference API")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (None for failed texts)
        """
        if not texts:
            return []
            
        try:
            # Filter out empty texts
            valid_texts = [t for t in texts if t and t.strip()]
            if not valid_texts:
                return [None] * len(texts)
            
            # Use Pinecone's inference API for batch embedding
            result = self.pc.inference.embed(
                model=self.embedding_model,
                inputs=valid_texts,
                parameters={"input_type": "passage"}
            )
            
            if result and result.data:
                embeddings = [item["values"] for item in result.data]
                logger.info(f"Generated {len(embeddings)} embeddings in batch")
                return embeddings
            else:
                logger.error("No embeddings returned from Pinecone inference API")
                return [None] * len(texts)
                
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [None] * len(texts)


def embed_case_node(case_data: dict, embedding_gen: EmbeddingGenerator) -> dict:
    """
    Add embeddings to case node data.
    
    Args:
        case_data: Case node data dictionary
        embedding_gen: EmbeddingGenerator instance
        
    Returns:
        Case data with embeddings added
    """
    if "summary" in case_data and case_data["summary"]:
        case_data["summary_embedding"] = embedding_gen.generate_embedding(case_data["summary"])
    
    if "outcome" in case_data and case_data["outcome"]:
        case_data["outcome_embedding"] = embedding_gen.generate_embedding(case_data["outcome"])
    
    return case_data


def embed_statute_node(statute_data: dict, embedding_gen: EmbeddingGenerator) -> dict:
    """
    Add embeddings to statute node data.
    
    Args:
        statute_data: Statute node data dictionary
        embedding_gen: EmbeddingGenerator instance
        
    Returns:
        Statute data with embeddings added
    """
    if "summary" in statute_data and statute_data["summary"]:
        statute_data["summary_embedding"] = embedding_gen.generate_embedding(statute_data["summary"])
    
    return statute_data


def embed_argument_node(argument_data: dict, embedding_gen: EmbeddingGenerator) -> dict:
    """
    Add embeddings to argument node data.
    
    Args:
        argument_data: Argument node data dictionary
        embedding_gen: EmbeddingGenerator instance
        
    Returns:
        Argument data with embeddings added
    """
    if "summary" in argument_data and argument_data["summary"]:
        argument_data["summary_embedding"] = embedding_gen.generate_embedding(argument_data["summary"])
    
    if "judge_reasoning" in argument_data and argument_data["judge_reasoning"]:
        argument_data["judge_reasoning_embedding"] = embedding_gen.generate_embedding(
            argument_data["judge_reasoning"]
        )
    
    return argument_data

