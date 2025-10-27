"""
Vector search tool using Pinecone for semantic search and chunk retrieval.
"""

import logging
from typing import List, Union, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from src.config import config
from src.models.schemas import Source

logger = logging.getLogger(__name__)


class VectorSearchTool:
    """
    Tool for searching and retrieving documents from Pinecone vector database.
    Supports both semantic search and direct chunk ID fetching.
    Uses Pinecone's inference API for automatic embedding generation.
    """

    def __init__(self):
        """Initialize Pinecone client and connect to index."""
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        self.namespace = config.PINECONE_NAMESPACE

        # Connect to existing index
        try:
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {e}")
            raise
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Union[Dict[str, Any], None] = None
    ) -> List[Source]:
        """
        Perform semantic search using Pinecone's inference API with llama-text-embed-v2.
        Pinecone automatically generates embeddings when using the inference API.

        Args:
            query: Natural language query string
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of Source objects with type='vector'
        """
        try:
            logger.info(f"Performing semantic search for query: {query[:100]}...")

            # Use Pinecone's inference API to generate embeddings and query
            # First, generate the embedding using the inference API
            from pinecone import Pinecone
            pc = Pinecone(api_key=config.PINECONE_API_KEY)

            # Generate embedding using Pinecone's inference API
            embedding_response = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=[query],
                parameters={"input_type": "query"}
            )

            # Extract the embedding vector
            query_embedding = embedding_response.data[0].values

            # Query Pinecone with the generated embedding
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
                filter=filter_dict
            )

            sources = []
            for match in results.get('matches', []):
                source = Source(
                    type="vector",
                    chunk_id=match['id'],
                    data=match.get('metadata', {}).get('chunk_text', ''),
                    metadata={
                        'score': match.get('score', 0.0),
                        'case_extract_name': match.get('metadata', {}).get('case_extract_name', ''),
                        **match.get('metadata', {})
                    }
                )
                sources.append(source)

            logger.info(f"Found {len(sources)} results from semantic search")
            return sources

        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []
    
    def fetch_by_ids(self, chunk_ids: List[str]) -> List[Source]:
        """
        Fetch specific chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of Source objects with type='vector'
        """
        try:
            logger.info(f"Fetching {len(chunk_ids)} chunks by ID")
            
            # Fetch vectors by ID
            results = self.index.fetch(ids=chunk_ids, namespace=self.namespace)
            
            sources = []
            for chunk_id, vector_data in results.get('vectors', {}).items():
                source = Source(
                    type="vector",
                    chunk_id=chunk_id,
                    data=vector_data.get('metadata', {}).get('chunk_text', ''),
                    metadata={
                        'case_extract_name': vector_data.get('metadata', {}).get('case_extract_name', ''),
                        **vector_data.get('metadata', {})
                    }
                )
                sources.append(source)
            
            logger.info(f"Successfully fetched {len(sources)} chunks")
            return sources
            
        except Exception as e:
            logger.error(f"Error fetching chunks by ID: {e}")
            return []
    
    def search(
        self,
        query: Union[str, None] = None,
        fetch_ids: Union[List[str], None] = None,
        top_k: int = 5
    ) -> List[Source]:
        """
        Main search method that routes to semantic search or fetch by ID.
        
        Args:
            query: Natural language query for semantic search
            fetch_ids: Specific chunk IDs to fetch
            top_k: Number of results for semantic search
            
        Returns:
            List of Source objects
        """
        if fetch_ids:
            return self.fetch_by_ids(fetch_ids)
        elif query:
            return self.semantic_search(query, top_k=top_k)
        else:
            logger.warning("No query or fetch_ids provided to vector search")
            return []


# Singleton instance
_vector_search_tool = None


def get_vector_search_tool() -> VectorSearchTool:
    """Get or create the singleton VectorSearchTool instance."""
    global _vector_search_tool
    if _vector_search_tool is None:
        _vector_search_tool = VectorSearchTool()
    return _vector_search_tool

