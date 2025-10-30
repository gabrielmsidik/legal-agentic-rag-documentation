"""
Reranking module for combining and reranking results from vector and graph searches.
Uses Pinecone's reranking API to intelligently rank all retrieved sources.
"""

import logging
from typing import List, Dict, Any, Union
from pinecone import Pinecone
from src.models.schemas import Source
from src.config import config

logger = logging.getLogger(__name__)


class RerankerTool:
    """
    Tool for reranking retrieved sources from both vector and graph databases.
    Uses Pinecone's inference API for intelligent reranking.
    """
    
    def __init__(self):
        """Initialize the reranker with Pinecone client."""
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.rerank_model = "bge-reranker-v2-m3"  # Multilingual reranker
        logger.info(f"Initialized RerankerTool with model: {self.rerank_model}")
    
    def _extract_text_for_reranking(self, source: Source) -> str:
        """
        Extract text content from a Source object for reranking.
        
        Args:
            source: Source object from vector or graph search
            
        Returns:
            String representation of the source content
        """
        if source.type == "vector":
            # For vector sources, use the text data directly
            if isinstance(source.data, str):
                return source.data
            else:
                return str(source.data)
        
        elif source.type == "graph":
            # For graph sources, extract key information
            if isinstance(source.data, dict):
                parts = []
                
                # Extract key fields in order of importance
                if "case_number" in source.data:
                    parts.append(f"Case: {source.data['case_number']}")
                if "name" in source.data:
                    parts.append(f"Name: {source.data['name']}")
                if "summary" in source.data:
                    parts.append(f"Summary: {source.data['summary']}")
                if "outcome" in source.data:
                    parts.append(f"Outcome: {source.data['outcome']}")
                if "section" in source.data:
                    parts.append(f"Section: {source.data['section']}")
                
                return " | ".join(parts) if parts else str(source.data)
            else:
                return str(source.data)
        
        return str(source.data)
    
    def rerank_sources(
        self,
        query: str,
        sources: List[Source],
        top_n: int = 5
    ) -> List[Source]:
        """
        Rerank all retrieved sources using Pinecone's reranking API.

        Args:
            query: The original user query
            sources: List of Source objects from vector and graph searches
            top_n: Number of top sources to return

        Returns:
            List of reranked Source objects, limited to top_n
        """
        if not sources:
            logger.warning("No sources provided for reranking")
            return []

        if len(sources) <= top_n:
            logger.info(f"Only {len(sources)} sources provided, returning all (no reranking needed)")
            return sources

        try:
            # Analyze source composition before reranking
            vector_count = sum(1 for s in sources if s.type == "vector")
            graph_count = sum(1 for s in sources if s.type == "graph")

            logger.info(f"=== RERANKING IMPACT ANALYSIS ===")
            logger.info(f"Query: {query[:100]}...")
            logger.info(f"Total sources before reranking: {len(sources)}")
            logger.info(f"  - Vector sources: {vector_count}")
            logger.info(f"  - Graph sources: {graph_count}")
            logger.info(f"Target: Top {top_n} sources after reranking")

            # Extract text content from all sources
            documents = [
                {
                    "id": str(i),
                    "text": self._extract_text_for_reranking(source)
                }
                for i, source in enumerate(sources)
            ]

            logger.debug(f"Extracted text from {len(documents)} sources for reranking")

            # Call Pinecone reranking API
            logger.info(f"Calling Pinecone reranking API (model: {self.rerank_model})...")
            rerank_result = self.pc.inference.rerank(
                model=self.rerank_model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=False
            )

            # Extract reranked indices and scores
            reranked_indices = []
            rerank_scores = {}

            for item in rerank_result.data:
                idx = int(item["index"])
                score = item["score"]
                reranked_indices.append(idx)
                rerank_scores[idx] = score

            # Analyze reranking impact
            reranked_vector_count = sum(1 for idx in reranked_indices if sources[idx].type == "vector")
            reranked_graph_count = sum(1 for idx in reranked_indices if sources[idx].type == "graph")

            # Calculate score statistics
            scores = list(rerank_scores.values())
            avg_score = sum(scores) / len(scores) if scores else 0
            min_score = min(scores) if scores else 0
            max_score = max(scores) if scores else 0

            logger.info(f"=== RERANKING RESULTS ===")
            logger.info(f"Sources selected: {len(reranked_indices)} (from {len(sources)} total)")
            logger.info(f"Reduction: {len(sources) - len(reranked_indices)} sources filtered out ({((len(sources) - len(reranked_indices)) / len(sources) * 100):.1f}%)")
            logger.info(f"Final composition:")
            logger.info(f"  - Vector sources: {reranked_vector_count} (was {vector_count})")
            logger.info(f"  - Graph sources: {reranked_graph_count} (was {graph_count})")
            logger.info(f"Rerank scores:")
            logger.info(f"  - Average: {avg_score:.4f}")
            logger.info(f"  - Min: {min_score:.4f}")
            logger.info(f"  - Max: {max_score:.4f}")
            logger.info(f"  - Range: {max_score - min_score:.4f}")

            # Log individual source rankings
            logger.info(f"=== TOP {top_n} RANKED SOURCES ===")
            for rank, idx in enumerate(reranked_indices, 1):
                source = sources[idx]
                score = rerank_scores[idx]
                source_type = source.type.upper()
                logger.info(f"Rank {rank}: {source_type} source (score: {score:.4f})")

            # Return sources in reranked order with scores added to metadata
            reranked_sources = []
            for idx in reranked_indices:
                source = sources[idx]
                # Add rerank score to metadata
                if source.metadata is None:
                    source.metadata = {}
                source.metadata["rerank_score"] = rerank_scores[idx]
                reranked_sources.append(source)

            logger.info(f"=== RERANKING COMPLETE ===")
            logger.info(f"Successfully reranked and selected top {len(reranked_sources)} sources")

            return reranked_sources

        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            logger.warning(f"Falling back to original source order (top {top_n})")
            logger.warning(f"Returning first {top_n} sources without reranking")
            # Fallback: return sources as-is if reranking fails
            return sources[:top_n]


# Singleton instance
_reranker = None


def get_reranker() -> RerankerTool:
    """Get or create the singleton RerankerTool instance."""
    global _reranker
    if _reranker is None:
        _reranker = RerankerTool()
    return _reranker

