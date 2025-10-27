"""
Graph search tool using Neo4j for structured knowledge graph queries.
Uses direct LLM-based natural language to Cypher conversion.
Supports both structured graph search and semantic embedding-based search.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import config
from src.models.schemas import Source
from src.tools.embedding_utils import EmbeddingGenerator

logger = logging.getLogger(__name__)


class GraphSearchTool:
    """
    Tool for querying Neo4j knowledge graph using natural language.
    Converts natural language queries to Cypher and executes them.
    """
    
    # Graph schema definition for the LLM
    GRAPH_SCHEMA = """
    Node Labels:
    - Case: Represents a legal case
      Properties: case_number (string), name (string), date (date), year (integer), summary (string), outcome (string)
    - Person: Represents a person (judge, lawyer, party)
      Properties: name (string), role (string)
    - Statute: Represents a legal statute
      Properties: name (string), section (string)
    - LegalPrinciple: Represents a legal principle or case type
      Properties: name (string)
    - Argument: Represents an argument made in a case
      Properties: summary (string), judge_reasoning (string), outcome (string), section (string), vector_chunk_ids (list[string])
    
    Relationships:
    - (Person)-[:PRESIDED_OVER]->(Case)
    - (Person)-[:IS_PARTY_IN {objective: string}]->(Case)
    - (Person)-[:REPRESENTED]->(Person)
    - (Case)-[:CITES]->(Case)
    - (Case)-[:CITES]->(Statute)
    - (Case)-[:CONTAINS]->(Argument)
    - (Case)-[:IS_RELATED_TO]->(LegalPrinciple)
    - (Person)-[:MADE]->(Argument)
    - (Argument)-[:IS_ABOUT]->(LegalPrinciple)
    - (Argument)-[:REFERENCES]->(Case)
    - (Argument)-[:REFERENCES]->(Statute)
    """
    
    def __init__(self):
        """Initialize Neo4j connection and LLM for Cypher generation."""
        self.uri = config.NEO4J_URI
        self.username = config.NEO4J_USERNAME
        self.password = config.NEO4J_PASSWORD
        self.database = config.NEO4J_DATABASE

        # Initialize Neo4j driver
        try:
            logger.info(f"Attempting to connect to Neo4j at: {self.uri}")
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

        # Initialize LLM for Cypher generation
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY
        )

        # Initialize embedding generator for semantic search
        self.embedding_gen = EmbeddingGenerator()

        # Semantic search configuration
        self.similarity_threshold = 0.5
        self.semantic_top_k = 5

        logger.info("Initialized GraphSearchTool with direct Cypher execution and semantic search")
    
    def execute_cypher(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query against Neo4j.
        
        Args:
            cypher_query: Cypher query string
            
        Returns:
            List of result records as dictionaries
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]
                logger.info(f"Executed Cypher query, got {len(records)} results")
                return records
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return []
    
    def _generate_cypher_query(self, natural_language_query: str) -> str:
        """
        Use LLM to generate a Cypher query from natural language.

        Args:
            natural_language_query: User's natural language query

        Returns:
            Generated Cypher query string
        """
        system_prompt = f"""You are an expert at converting natural language questions into Cypher queries for Neo4j.

            Given the following graph schema:
            {self.GRAPH_SCHEMA}

            Generate a Cypher query that answers the user's question. Follow these rules:
            1. Return ONLY the Cypher query, no explanations or markdown
            2. Use MATCH clauses to find relevant nodes and relationships
            3. Use WHERE clauses for filtering with case-insensitive matching
            4. Return relevant properties and relationships
            5. Limit results to 10 unless otherwise specified
            6. For questions about cases, include case_number, name, date, summary, and outcome
            7. For questions about arguments, include summary, judge_reasoning, outcome, and vector_chunk_ids
            8. Use OPTIONAL MATCH for relationships that might not exist
            9. IMPORTANT: Use case-insensitive matching for text properties (toLower() function)
            10. Use CONTAINS for partial matching instead of exact equality
            11. If searching for legal principles, use toLower() for case-insensitive comparison
            12. Try multiple search strategies: by legal principle, by argument content, by case summary

            Example queries:
            - "Find cases about custody" -> MATCH (c:Case)-[:IS_RELATED_TO]->(lp:LegalPrinciple) WHERE toLower(lp.name) CONTAINS 'custody' RETURN c.case_number, c.name, c.date, c.summary LIMIT 10
            - "Who presided over case [2024] SGFC 1?" -> MATCH (p:Person)-[:PRESIDED_OVER]->(c:Case) WHERE c.case_number CONTAINS '[2024] SGFC 1' RETURN p.name, p.role
            - "Find arguments about maintenance" -> MATCH (a:Argument)-[:IS_ABOUT]->(lp:LegalPrinciple) WHERE toLower(lp.name) CONTAINS 'maintenance' OR toLower(a.summary) CONTAINS 'maintenance' RETURN a.summary, a.judge_reasoning, a.vector_chunk_ids LIMIT 10
            - "Find cases about joint accounts" -> MATCH (c:Case) WHERE toLower(c.summary) CONTAINS 'joint' OR toLower(c.summary) CONTAINS 'account' OPTIONAL MATCH (c)-[:CONTAINS]->(a:Argument) WHERE toLower(a.summary) CONTAINS 'joint' OR toLower(a.summary) CONTAINS 'account' RETURN c.case_number, c.name, c.summary, a.summary, a.judge_reasoning LIMIT 10
        """

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=natural_language_query)
            ]

            response = self.llm.invoke(messages)
            cypher_query = response.content.strip()

            # Remove markdown code blocks if present
            if cypher_query.startswith("```"):
                lines = cypher_query.split("\n")
                cypher_query = "\n".join(lines[1:-1]) if len(lines) > 2 else cypher_query
                cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()

            logger.info(f"Generated Cypher query: {cypher_query}")
            return cypher_query

        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
            return ""

    def natural_language_search(self, query: str) -> List[Source]:
        """
        Convert natural language query to Cypher and execute it.

        Args:
            query: Natural language query

        Returns:
            List of Source objects with type='graph'
        """
        try:
            logger.info(f"Processing natural language query: {query[:100]}...")

            # Generate Cypher query using LLM
            generated_cypher = self._generate_cypher_query(query)

            if not generated_cypher:
                logger.warning("Failed to generate Cypher query")
                return []

            # Execute the Cypher query
            raw_results = self.execute_cypher(generated_cypher)

            # Convert results to Source objects
            sources = []
            for idx, record in enumerate(raw_results):
                # Extract vector_chunk_ids if present in the result
                chunk_ids = []
                for key, value in record.items():
                    if isinstance(value, dict) and 'vector_chunk_ids' in value:
                        chunk_ids = value['vector_chunk_ids']
                    elif key == 'vector_chunk_ids':
                        chunk_ids = value if isinstance(value, list) else []

                source = Source(
                    type="graph",
                    chunk_id=None,
                    data=record,
                    metadata={
                        'generated_cypher': generated_cypher,
                        'vector_chunk_ids': chunk_ids,
                        'result_index': idx
                    }
                )
                sources.append(source)

            logger.info(f"Found {len(sources)} graph results")
            return sources

        except Exception as e:
            logger.error(f"Error during natural language graph search: {e}")
            return []
    
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for query text.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector or None if generation fails
        """
        try:
            embedding = self.embedding_gen.generate_embedding(query)
            if embedding:
                logger.debug(f"Generated query embedding of dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None

    def _search_by_embedding(
        self,
        node_label: str,
        embedding_field: str,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Source]:
        """
        Execute vector similarity search on a specific node type.

        Args:
            node_label: Neo4j node label (e.g., 'Case', 'Argument', 'Statute')
            embedding_field: Property name containing embedding (e.g., 'summary_embedding')
            query_embedding: Query embedding vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of Source objects with similarity scores
        """
        try:
            # Build Cypher query for vector similarity search
            cypher_query = f"""
            MATCH (n:{node_label})
            WHERE n.`{embedding_field}` IS NOT NULL
            WITH n,
                 vector.similarity.cosine($query_embedding, n.`{embedding_field}`) AS similarity
            WHERE similarity > $threshold
            RETURN n, similarity
            ORDER BY similarity DESC
            LIMIT $top_k
            """

            logger.debug(f"Executing semantic search on {node_label}.{embedding_field}")

            with self.driver.session(database=self.database) as session:
                result = session.run(
                    cypher_query,
                    query_embedding=query_embedding,
                    threshold=similarity_threshold,
                    top_k=top_k
                )
                records = [record.data() for record in result]

            # Convert results to Source objects
            sources = []
            for idx, record in enumerate(records):
                node = record.get('n')
                similarity = record.get('similarity', 0)

                if node:
                    # Extract node properties
                    node_data = dict(node)

                    source = Source(
                        type="graph",
                        chunk_id=None,
                        data=node_data,
                        metadata={
                            'node_label': node_label,
                            'embedding_field': embedding_field,
                            'similarity_score': similarity,
                            'search_type': 'semantic',
                            'result_index': idx
                        }
                    )
                    sources.append(source)

            logger.info(f"Semantic search on {node_label}.{embedding_field} returned {len(sources)} results")
            return sources

        except Exception as e:
            logger.error(f"Error during semantic search on {node_label}: {e}")
            return []

    def semantic_graph_search(self, query: str, top_k: int = 5) -> List[Source]:
        """
        Search Neo4j using embedding similarity on node properties.
        Searches across multiple node types and embedding fields.

        Args:
            query: Natural language query
            top_k: Number of results per node type

        Returns:
            List of Source objects from semantic search
        """
        logger.info(f"Performing semantic graph search for query: {query[:100]}...")

        # Generate embedding for query
        query_embedding = self._get_query_embedding(query)
        if not query_embedding:
            logger.warning("Failed to generate query embedding, returning empty results")
            return []

        all_results = []

        # Search Case nodes by summary_embedding
        case_results = self._search_by_embedding(
            node_label="Case",
            embedding_field="summary_embedding",
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=self.similarity_threshold
        )
        all_results.extend(case_results)

        # Search Case nodes by outcome_embedding
        case_outcome_results = self._search_by_embedding(
            node_label="Case",
            embedding_field="outcome_embedding",
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=self.similarity_threshold
        )
        all_results.extend(case_outcome_results)

        # Search Argument nodes by summary_embedding
        argument_results = self._search_by_embedding(
            node_label="Argument",
            embedding_field="summary_embedding",
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=self.similarity_threshold
        )
        all_results.extend(argument_results)

        # Search Argument nodes by judge_reasoning_embedding
        argument_reasoning_results = self._search_by_embedding(
            node_label="Argument",
            embedding_field="judge_reasoning_embedding",
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=self.similarity_threshold
        )
        all_results.extend(argument_reasoning_results)

        # Search Statute nodes by summary_embedding
        statute_results = self._search_by_embedding(
            node_label="Statute",
            embedding_field="summary_embedding",
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=self.similarity_threshold
        )
        all_results.extend(statute_results)

        # Deduplicate results by node identity
        deduplicated = self._deduplicate_results(all_results)

        logger.info(f"Semantic graph search returned {len(deduplicated)} unique results")
        return deduplicated

    def _deduplicate_results(self, results: List[Source]) -> List[Source]:
        """
        Remove duplicate results and keep highest similarity score.

        Args:
            results: List of Source objects

        Returns:
            Deduplicated list of Source objects
        """
        seen = {}
        for result in results:
            # Use node properties as key to identify duplicates
            node_data = result.data

            # Create a unique key based on node type and identifying properties
            if 'case_number' in node_data:
                key = ('Case', node_data['case_number'])
            elif 'name' in node_data and 'section' in node_data:
                key = ('Statute', node_data['name'], node_data['section'])
            elif 'summary' in node_data:
                # For arguments, use summary as part of key
                key = ('Argument', node_data.get('summary', '')[:50])
            else:
                # Fallback: use all properties
                key = tuple(sorted(node_data.items()))

            # Keep result with highest similarity score
            if key not in seen:
                seen[key] = result
            else:
                current_score = result.metadata.get('similarity_score', 0)
                existing_score = seen[key].metadata.get('similarity_score', 0)
                if current_score > existing_score:
                    seen[key] = result

        return list(seen.values())

    def search(self, query: str, use_semantic: bool = False) -> List[Source]:
        """
        Main search method for graph queries.

        Args:
            query: Natural language query
            use_semantic: If True, use semantic search; if False, use structured search

        Returns:
            List of Source objects
        """
        if use_semantic:
            return self.semantic_graph_search(query)
        else:
            return self.natural_language_search(query)
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")


# Singleton instance
_graph_search_tool = None


def get_graph_search_tool() -> GraphSearchTool:
    """Get or create the singleton GraphSearchTool instance."""
    global _graph_search_tool
    if _graph_search_tool is None:
        _graph_search_tool = GraphSearchTool()
    return _graph_search_tool

