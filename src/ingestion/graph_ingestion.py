"""
Graph ingestion pipeline for extracting entities and relationships from legal cases
and storing them in Neo4j knowledge graph.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from openai import OpenAI
from pinecone import Pinecone
from src.config import config
from src.models.schemas import (
    ExtractionResult, CaseNode, PersonNode, StatuteNode, LegalPrincipleNode, ArgumentNode,
    FirstPassExtractionResult, SecondPassExtractionResult,
    CaseNodeForOpenAI, PersonNodeForOpenAI, LegalPrincipleNodeForOpenAI,
    ArgumentNodeForOpenAI, StatuteNodeForOpenAI,
    PersonRole, LegalPrincipleType, ArgumentOutcome
)
from src.tools.embedding_utils import EmbeddingGenerator, embed_case_node, embed_statute_node, embed_argument_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def robust_json_parse(content: str, max_retries: int = 5) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON from LLM response with multiple fallback strategies.
    Handles truncated/incomplete JSON by attempting to fix common issues.

    Args:
        content: Raw content from LLM response
        max_retries: Number of parsing strategies to try

    Returns:
        Parsed JSON dict or None if all strategies fail
    """
    def fix_truncated_json(c: str) -> str:
        """Attempt to fix truncated JSON by closing open braces/brackets."""
        # Count open and close braces
        open_braces = c.count('{') - c.count('}')
        open_brackets = c.count('[') - c.count(']')

        # Add missing closing braces and brackets
        fixed = c.rstrip()
        if fixed.endswith(','):
            fixed = fixed[:-1]  # Remove trailing comma

        fixed += ']' * open_brackets
        fixed += '}' * open_braces
        return fixed

    strategies = [
        # Strategy 1: Direct parse after stripping whitespace
        lambda c: json.loads(c.strip()),

        # Strategy 2: Remove markdown code blocks
        lambda c: json.loads(
            re.sub(r'^```(?:json)?\s*\n?', '',
                   re.sub(r'\n?```\s*$', '', c.strip()))
        ),

        # Strategy 3: Extract JSON from anywhere in the text
        lambda c: json.loads(
            re.search(r'\{.*\}', c, re.DOTALL).group(0)
        ) if re.search(r'\{.*\}', c, re.DOTALL) else None,

        # Strategy 4: Fix common JSON issues (trailing commas, unquoted keys)
        lambda c: json.loads(
            re.sub(r',(\s*[}\]])', r'\1',  # Remove trailing commas
                   re.sub(r'^```(?:json)?\s*\n?', '',
                          re.sub(r'\n?```\s*$', '', c.strip())))
        ),

        # Strategy 5: Fix truncated JSON by closing open braces/brackets
        lambda c: json.loads(fix_truncated_json(c)),
    ]

    for i, strategy in enumerate(strategies[:max_retries], 1):
        try:
            result = strategy(content)
            if result is not None:
                logger.debug(f"JSON parsed successfully using strategy {i}")
                return result
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.debug(f"Strategy {i} failed: {e}")
            continue

    return None


class GraphIngestionPipeline:
    """
    Pipeline for ingesting legal case data into Neo4j knowledge graph.
    Performs entity and relationship extraction using LLM.
    """
    
    def __init__(self):
        """Initialize connections to Pinecone, Neo4j, LLM, and embedding generator."""
        # Initialize Pinecone
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        self.namespace = config.PINECONE_NAMESPACE

        # Initialize Neo4j
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        self.database = config.NEO4J_DATABASE

        # Initialize OpenAI client for beta API
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

        # Initialize embedding generator
        self.embedding_gen = EmbeddingGenerator()

        logger.info("GraphIngestionPipeline initialized")
    
    def fetch_case_extract_from_pinecone(self, case_extract_name: str) -> str:
        """
        Fetch all chunks for a case extract from Pinecone and concatenate them.
        Uses metadata filter with query() for efficient retrieval.

        Args:
            case_extract_name: The case extract identifier (e.g., "case_extract_2024_1")

        Returns:
            Concatenated text of all chunks and the chunks as a list of enums
        """
        logger.info(f"Fetching case extract: {case_extract_name}")

        chunks = []

        try:
            # Use query with metadata filter - more efficient than dummy vector
            # The filter does the heavy lifting, not the vector similarity
            results = self.index.query(
                vector=[0] * 1024,  # Dummy vector (required by API)
                top_k=10000,  # Maximum allowed
                namespace=self.namespace,
                filter={"case_extract_name": {"$eq": case_extract_name}},
                include_metadata=True
            )

            # Extract chunks
            for match in results.get('matches', []):
                chunk_id = match['id']
                chunk_text = match.get('metadata', {}).get('chunk_text', '')
                chunks.append((chunk_id, chunk_text))

            # Sort by chunk ID to maintain order
            chunks.sort(key=lambda x: x[0])

            # Concatenate
            full_text = "\n\n".join([text for _, text in chunks])

            logger.info(f"Fetched {len(chunks)} chunks for {case_extract_name}")

            return full_text, chunks

        except Exception as e:
            logger.error(f"Error fetching case extract {case_extract_name}: {e}")
            return "", []
    
    def extract_first_pass(self, case_text: str, case_number: str) -> Dict[str, Any]:
        """
        First pass extraction: Case, Person, and LegalPrinciple nodes.
        Uses OpenAI Structured Output API to enforce Pydantic schema.

        Args:
            case_text: Full text of the case
            case_number: Case number (e.g., "2024_SGHCF_1")

        Returns:
            Dictionary with extracted entities
        """
        logger.info(f"Performing first pass extraction for {case_number}")

        system_prompt = """You are a legal entity extraction expert. Extract the following information from the case text:

1. Case information: case_number, name, date, year, summary, outcome
2. All persons mentioned: judges, lawyers, parties (appellant, respondent, etc.)
3. All legal principles or case types mentioned

For persons:
- For judges and lawyers: use canonical names (e.g., "Tan Lee Meng" not "Justice Tan Lee Meng")
- For parties: create composite IDs like "XDJ-Appellant-2024_SGHCF_1" if the party is anonymized
- For role: use one of the provided enum values (Judge, Appellant, Respondent, Lawyer for Appellant, Lawyer for Respondent, Witness, Mediator, Others)
- If role doesn't fit these categories, use "Others" and provide details in role_others field

For legal principles:
- For name: use one of the provided enum values (Division of Matrimonial Assets, Custody, Care and Control, Maintenance and Support, Guardianship, Adoption, Ancillary Matters, Family Violence, Matrimonial Property, Child Protection, Succession, Others)
- If principle doesn't fit these categories, use "Others" and provide details in name_others field

Extract all entities and return them in the specified structured format."""

        user_prompt = f"""Case Number: {case_number}

Case Text:
{case_text[:8000]}

Extract the entities as specified."""

        try:
            # Use OpenAI beta API with Pydantic model
            response = self.openai_client.beta.chat.completions.parse(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=FirstPassExtractionResult
            )

            # Get the parsed result directly
            result = response.choices[0].message.parsed

            logger.info(f"First pass extraction completed: {len(result.persons)} persons, {len(result.legal_principles)} principles")

            # Add optional fields back for storage
            case_data = result.case.model_dump()
            persons_data = []
            for p in result.persons:
                p_dict = p.model_dump()
                p_dict["role_others"] = None  # Add optional field
                persons_data.append(p_dict)

            legal_principles_data = []
            for lp in result.legal_principles:
                lp_dict = lp.model_dump()
                lp_dict["name_others"] = None  # Add optional field
                legal_principles_data.append(lp_dict)

            return {
                "case": case_data,
                "persons": persons_data,
                "legal_principles": legal_principles_data
            }

        except Exception as e:
            logger.error(f"Failed to extract first pass for {case_number}: {e}")
            return {"case": {}, "persons": [], "legal_principles": []}
    
    def extract_second_pass(self, case_text: str, chunks: List[tuple], case_number: str) -> Dict[str, Any]:
        """
        Second pass extraction: Argument nodes and relationships.
        Processes chunks iteratively for better context and logging.
        Uses OpenAI Structured Output API to enforce Pydantic schema.

        Args:
            case_text: Full text of the case (for fallback)
            chunks: List of (chunk_id, chunk_text) tuples
            case_number: Case number

        Returns:
            Dictionary with extracted arguments and relationships
        """
        logger.info(f"Performing second pass extraction for {case_number}")
        logger.info(f"Processing {len(chunks)} chunks iteratively")

        system_prompt = """You are a legal reasoning extraction expert. Extract all arguments made in the case.

            For each argument, extract:
            - summary: 1-2 sentence summary of the party's claim
            - judge_reasoning: 1-2 sentence summary of judge's evaluation
            - outcome: Use one of the provided enum values (Accepted, Rejected, Partially Accepted, Dismissed, Allowed, Dismissed with Costs, Others)
            - section: The section where this is discussed
            - vector_chunk_ids: List of chunk IDs (format: case_extract_YYYY_N_section)
            - related_legal_principles: Names of legal principles this argument relates to
            - referenced_statutes: List of {"name": "...", "section": "..."}
            - referenced_cases: List of case numbers

            For outcome: use one of the provided enum values. If outcome doesn't fit, use "Others" and provide details in outcome_others field.

            Extract all arguments and return them in the specified structured format."""

        all_arguments = []
        all_statutes = []
        all_cases = set()

        # Process chunks iteratively
        for chunk_idx, (chunk_id, chunk_text) in enumerate(chunks, 1):
            logger.info(f"Processing chunk {chunk_idx}/{len(chunks)}: {chunk_id}")

            if not chunk_text or not chunk_text.strip():
                logger.debug(f"Skipping empty chunk: {chunk_id}")
                continue

            user_prompt = f"""Case Number: {case_number}
Chunk ID: {chunk_id}

Case Text (Chunk):
{chunk_text}

Extract all arguments from this chunk and their relationships."""

            try:
                # Use OpenAI beta API with Pydantic model
                response = self.openai_client.beta.chat.completions.parse(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=SecondPassExtractionResult
                )

                # Get the parsed result directly
                result = response.choices[0].message.parsed

                # Process arguments from this chunk
                chunk_arguments = 0
                for arg in result.arguments:
                    arg_dict = arg.model_dump()
                    arg_dict["outcome_others"] = None  # Add optional field
                    arg_dict["chunk_id"] = chunk_id  # Track which chunk this came from
                    all_arguments.append(arg_dict)
                    chunk_arguments += 1

                # Process statutes from this chunk
                chunk_statutes = 0
                for statute in result.statutes:
                    statute_dict = statute.model_dump()
                    statute_dict["chunk_id"] = chunk_id  # Track which chunk this came from
                    all_statutes.append(statute_dict)
                    chunk_statutes += 1

                # Collect referenced cases
                for case_ref in result.referenced_cases:
                    all_cases.add(case_ref)

                # Log chunk completion
                logger.info(f"✅ Chunk {chunk_idx} ({chunk_id}): {chunk_arguments} arguments, {chunk_statutes} statutes extracted and persisted")

            except Exception as e:
                logger.error(f"❌ Failed to extract from chunk {chunk_id}: {e}")
                continue

        logger.info(f"Second pass extraction completed: {len(all_arguments)} total arguments, {len(all_statutes)} total statutes, {len(all_cases)} referenced cases")

        return {
            "arguments": all_arguments,
            "statutes": all_statutes,
            "cases": list(all_cases)
        }

    def ingest_to_neo4j(self, extraction_data: Dict[str, Any], case_number: str) -> None:
        """
        Ingest extracted data into Neo4j using MERGE for entities and CREATE for arguments.

        Args:
            extraction_data: Combined extraction data from both passes
            case_number: Case number
        """
        logger.info(f"Ingesting data to Neo4j for {case_number}")

        with self.driver.session(database=self.database) as session:
            # 1. Create/Merge Case node with embeddings
            case_data = extraction_data.get("case", {})
            if case_data:
                # Generate embeddings for semantic fields
                case_data = embed_case_node(case_data, self.embedding_gen)

                session.run("""
                    MERGE (c:Case {case_number: $case_number})
                    SET c.name = $name,
                        c.date = $date,
                        c.year = $year,
                        c.summary = $summary,
                        c.summary_embedding = $summary_embedding,
                        c.outcome = $outcome,
                        c.outcome_embedding = $outcome_embedding
                """, **case_data)
                logger.info(f"Created/Updated Case node: {case_number}")

            # 2. Create/Merge Person nodes
            for person in extraction_data.get("persons", []):
                session.run("""
                    MERGE (p:Person {name: $name})
                    SET p.role = $role
                """, **person)

                # Create relationships based on role
                role = person.get("role", "")
                if "Judge" in role or "judge" in role:
                    session.run("""
                        MATCH (p:Person {name: $name})
                        MATCH (c:Case {case_number: $case_number})
                        MERGE (p)-[:PRESIDED_OVER]->(c)
                    """, name=person["name"], case_number=case_number)
                elif "Appellant" in role or "Respondent" in role or "party" in role.lower():
                    session.run("""
                        MATCH (p:Person {name: $name})
                        MATCH (c:Case {case_number: $case_number})
                        MERGE (p)-[:IS_PARTY_IN]->(c)
                    """, name=person["name"], case_number=case_number)

            logger.info(f"Created/Updated {len(extraction_data.get('persons', []))} Person nodes")

            # 3. Create/Merge LegalPrinciple nodes
            for principle in extraction_data.get("legal_principles", []):
                session.run("""
                    MERGE (lp:LegalPrinciple {name: $name})
                """, **principle)

                # Link to case
                session.run("""
                    MATCH (lp:LegalPrinciple {name: $name})
                    MATCH (c:Case {case_number: $case_number})
                    MERGE (c)-[:IS_RELATED_TO]->(lp)
                """, name=principle["name"], case_number=case_number)

            logger.info(f"Created/Updated {len(extraction_data.get('legal_principles', []))} LegalPrinciple nodes")

            # 4. Create Argument nodes (always CREATE, not MERGE) with embeddings
            for arg in extraction_data.get("arguments", []):
                # Generate embeddings for semantic fields
                arg = embed_argument_node(arg, self.embedding_gen)

                # Create the argument node
                arg_id = session.run("""
                    CREATE (a:Argument {
                        summary: $summary,
                        summary_embedding: $summary_embedding,
                        judge_reasoning: $judge_reasoning,
                        judge_reasoning_embedding: $judge_reasoning_embedding,
                        outcome: $outcome,
                        section: $section,
                        vector_chunk_ids: $vector_chunk_ids
                    })
                    RETURN elementId(a) as arg_id
                """,
                    summary=arg.get("summary", ""),
                    summary_embedding=arg.get("summary_embedding"),
                    judge_reasoning=arg.get("judge_reasoning", ""),
                    judge_reasoning_embedding=arg.get("judge_reasoning_embedding"),
                    outcome=arg.get("outcome", ""),
                    section=arg.get("section", ""),
                    vector_chunk_ids=arg.get("vector_chunk_ids", [])
                ).single()["arg_id"]

                # Link argument to case
                session.run("""
                    MATCH (c:Case {case_number: $case_number})
                    MATCH (a:Argument) WHERE elementId(a) = $arg_id
                    MERGE (c)-[:CONTAINS]->(a)
                """, case_number=case_number, arg_id=arg_id)

                # Link argument to person who made it
                made_by_role = arg.get("made_by_role", "")
                if made_by_role:
                    session.run("""
                        MATCH (p:Person) WHERE p.role CONTAINS $role
                        MATCH (a:Argument) WHERE elementId(a) = $arg_id
                        WITH p, a LIMIT 1
                        MERGE (p)-[:MADE]->(a)
                    """, role=made_by_role, arg_id=arg_id)

                # Link argument to legal principles
                for principle_name in arg.get("legal_principles", []):
                    session.run("""
                        MERGE (lp:LegalPrinciple {name: $name})
                        WITH lp
                        MATCH (a:Argument) WHERE elementId(a) = $arg_id
                        MERGE (a)-[:IS_ABOUT]->(lp)
                    """, name=principle_name, arg_id=arg_id)

                # Link argument to statutes with embeddings
                for statute in arg.get("statutes", []):
                    # Generate embeddings for statute summary if available
                    statute_data = embed_statute_node(statute, self.embedding_gen)

                    session.run("""
                        MERGE (s:Statute {name: $name, section: $section})
                        SET s.summary = $summary,
                            s.summary_embedding = $summary_embedding
                        WITH s
                        MATCH (a:Argument) WHERE elementId(a) = $arg_id
                        MERGE (a)-[:REFERENCES]->(s)
                    """,
                        name=statute_data.get("name", ""),
                        section=statute_data.get("section", ""),
                        summary=statute_data.get("summary", ""),
                        summary_embedding=statute_data.get("summary_embedding"),
                        arg_id=arg_id)

                    # Also link case to statute
                    session.run("""
                        MATCH (c:Case {case_number: $case_number})
                        MATCH (s:Statute {name: $name, section: $section})
                        MERGE (c)-[:CITES]->(s)
                    """, case_number=case_number, name=statute_data.get("name", ""), section=statute_data.get("section", ""))

                # Link argument to referenced cases
                for ref_case in arg.get("cases", []):
                    session.run("""
                        MERGE (rc:Case {case_number: $ref_case_number})
                        WITH rc
                        MATCH (a:Argument) WHERE elementId(a) = $arg_id
                        MERGE (a)-[:REFERENCES]->(rc)
                    """, ref_case_number=ref_case, arg_id=arg_id)

                    # Also link case to referenced case
                    session.run("""
                        MATCH (c:Case {case_number: $case_number})
                        MATCH (rc:Case {case_number: $ref_case_number})
                        MERGE (c)-[:CITES]->(rc)
                    """, case_number=case_number, ref_case_number=ref_case)

            logger.info(f"Created {len(extraction_data.get('arguments', []))} Argument nodes")

    def ingest_case(self, case_extract_name: str) -> None:
        """
        Main ingestion method for a single case extract.

        Args:
            case_extract_name: Case extract identifier (e.g., "case_extract_2024_1")
        """
        logger.info(f"Starting ingestion for {case_extract_name}")

        # Extract case number from case_extract_name
        # Format: case_extract_YYYY_N
        parts = case_extract_name.split("_")
        if len(parts) >= 3:
            year = parts[2]
            case_num = parts[3] if len(parts) > 3 else "1"
            case_number = f"{year}_SGHCF_{case_num}"
        else:
            logger.error(f"Invalid case_extract_name format: {case_extract_name}")
            return

        # Step 1: Fetch case text from Pinecone
        case_text, chunks = self.fetch_case_extract_from_pinecone(case_extract_name)

        if not case_text:
            logger.error(f"No text found for {case_extract_name}")
            return

        # Step 2: First pass extraction
        first_pass = self.extract_first_pass(case_text, case_number)

        # Step 3: Second pass extraction
        second_pass = self.extract_second_pass(case_text, chunks, case_number)

        # Step 4: Combine results
        extraction_data = {
            "case": first_pass.get("case", {}),
            "persons": first_pass.get("persons", []),
            "legal_principles": first_pass.get("legal_principles", []),
            "arguments": second_pass.get("arguments", []),
            "statutes": second_pass.get("statutes", []),
            "cases": second_pass.get("cases", [])
        }

        # Step 5: Ingest to Neo4j
        self.ingest_to_neo4j(extraction_data, case_number)

        logger.info(f"Completed ingestion for {case_extract_name}")

    def close(self):
        """Close database connections."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")


def main():
    """Main function for running the ingestion pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest legal cases into Neo4j knowledge graph")
    parser.add_argument(
        "case_extract_name",
        type=str,
        help="Case extract name (e.g., 'case_extract_2024_1')"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = GraphIngestionPipeline()

    try:
        # Ingest the case
        pipeline.ingest_case(args.case_extract_name)
    finally:
        # Clean up
        pipeline.close()


if __name__ == "__main__":
    main()

