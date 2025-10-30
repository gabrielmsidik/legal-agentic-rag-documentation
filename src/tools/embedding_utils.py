"""
Utility functions for generating and managing embeddings for semantic search.
Supports two embedding models:
1. OpenAI's text-embedding-3-small (1536 dimensions) - for Neo4j semantic search
2. Pinecone's llama-text-embed-v2 (1024 dimensions) - for Pinecone vector search
"""

import logging
from typing import List, Optional
from openai import OpenAI
from pinecone import Pinecone
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for semantic search using two different models:
    1. OpenAI's text-embedding-3-small for Neo4j (1536 dimensions)
    2. Pinecone's llama-text-embed-v2 for Pinecone (1024 dimensions)
    """

    def __init__(self):
        """Initialize OpenAI and Pinecone clients for embedding generation."""
        # OpenAI client for Neo4j embeddings
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.openai_model = "text-embedding-3-small"
        self.openai_dimension = 1536

        # Pinecone client for Pinecone embeddings
        self.pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
        self.pinecone_model = "llama-text-embed-v2"
        self.pinecone_dimension = 1024

    def generate_embedding_for_neo4j(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for Neo4j using OpenAI's text-embedding-3-small model.
        Returns 1536-dimensional embeddings.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for Neo4j embedding generation")
            return None

        try:
            # Use OpenAI's embedding API
            response = self.openai_client.embeddings.create(
                model=self.openai_model,
                input=text
            )

            if response and response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                logger.debug(f"Generated Neo4j embedding of dimension {len(embedding)}")
                return embedding
            else:
                logger.error(f"No embedding returned from OpenAI API: {response}")
                return None

        except Exception as e:
            logger.error(f"Failed to generate Neo4j embedding: {e}")
            return None

    def generate_embedding_for_pinecone(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for Pinecone using Pinecone's inference API.
        Returns 1024-dimensional embeddings using llama-text-embed-v2 model.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for Pinecone embedding generation")
            return None

        try:
            # Use Pinecone's inference API for embedding
            response = self.pinecone_client.inference.embed(
                model=self.pinecone_model,
                inputs=[text],
                parameters={"input_type": "passage"}
            )

            if response and response.data and len(response.data) > 0:
                embedding = response.data[0]["values"]
                logger.debug(f"Generated Pinecone embedding of dimension {len(embedding)}")
                return embedding
            else:
                logger.error(f"No embedding returned from Pinecone API: {response}")
                return None

        except Exception as e:
            logger.error(f"Failed to generate Pinecone embedding: {e}")
            return None

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Legacy method for backward compatibility. Uses OpenAI embedding (Neo4j).

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        return self.generate_embedding_for_neo4j(text)

    def generate_embeddings_batch_for_neo4j(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batch using OpenAI's API.

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

            # Use OpenAI's embedding API for batch embedding
            response = self.openai_client.embeddings.create(
                model=self.openai_model,
                input=valid_texts
            )

            if response and response.data:
                # Sort by index to maintain order
                embeddings_dict = {item.index: item.embedding for item in response.data}
                embeddings = [embeddings_dict.get(i) for i in range(len(valid_texts))]
                logger.info(f"Generated {len(embeddings)} Neo4j embeddings in batch")
                return embeddings
            else:
                logger.error(f"No embeddings returned from OpenAI API: {response}")
                return [None] * len(texts)

        except Exception as e:
            logger.error(f"Failed to generate batch Neo4j embeddings: {e}")
            return [None] * len(texts)

    def generate_embeddings_batch_for_pinecone(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batch using Pinecone's inference API.

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
            response = self.pinecone_client.inference.embed(
                model=self.pinecone_model,
                inputs=valid_texts,
                parameters={"input_type": "passage"}
            )

            if response and response.data:
                embeddings = [item["values"] for item in response.data]
                logger.info(f"Generated {len(embeddings)} Pinecone embeddings in batch")
                return embeddings
            else:
                logger.error(f"No embeddings returned from Pinecone API: {response}")
                return [None] * len(texts)

        except Exception as e:
            logger.error(f"Failed to generate batch Pinecone embeddings: {e}")
            return [None] * len(texts)

    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Legacy method for backward compatibility. Uses OpenAI batch embedding (Neo4j).

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (None for failed texts)
        """
        return self.generate_embeddings_batch_for_neo4j(texts)


def embed_case_node(case_data: dict, embedding_gen: EmbeddingGenerator) -> dict:
    """
    Add embeddings to case node data using Neo4j embeddings (OpenAI).

    Args:
        case_data: Case node data dictionary
        embedding_gen: EmbeddingGenerator instance

    Returns:
        Case data with embeddings added
    """
    if "summary" in case_data and case_data["summary"]:
        case_data["summary_embedding"] = embedding_gen.generate_embedding_for_neo4j(case_data["summary"])

    if "outcome" in case_data and case_data["outcome"]:
        case_data["outcome_embedding"] = embedding_gen.generate_embedding_for_neo4j(case_data["outcome"])

    return case_data


def embed_statute_node(statute_data: dict, embedding_gen: EmbeddingGenerator) -> dict:
    """
    Add embeddings to statute node data using Neo4j embeddings (OpenAI).

    Args:
        statute_data: Statute node data dictionary
        embedding_gen: EmbeddingGenerator instance

    Returns:
        Statute data with embeddings added
    """
    if "summary" in statute_data and statute_data["summary"]:
        statute_data["summary_embedding"] = embedding_gen.generate_embedding_for_neo4j(statute_data["summary"])

    return statute_data


def embed_argument_node(argument_data: dict, embedding_gen: EmbeddingGenerator) -> dict:
    """
    Add embeddings to argument node data using Neo4j embeddings (OpenAI).

    Args:
        argument_data: Argument node data dictionary
        embedding_gen: EmbeddingGenerator instance

    Returns:
        Argument data with embeddings added
    """
    if "summary" in argument_data and argument_data["summary"]:
        argument_data["summary_embedding"] = embedding_gen.generate_embedding_for_neo4j(argument_data["summary"])

    if "judge_reasoning" in argument_data and argument_data["judge_reasoning"]:
        argument_data["judge_reasoning_embedding"] = embedding_gen.generate_embedding_for_neo4j(
            argument_data["judge_reasoning"]
        )

    return argument_data

