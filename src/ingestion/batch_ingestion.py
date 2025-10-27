"""
Batch ingestion script for loading all cases from Pinecone into Neo4j.

This script:
1. Queries Pinecone to find all unique case_extract_name values
2. Checks Neo4j to see which cases have already been loaded
3. Ingests only the cases that haven't been loaded yet
"""

import logging
from typing import Set, List
from pinecone import Pinecone
from neo4j import GraphDatabase

from src.config import config
from src.ingestion.graph_ingestion import GraphIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchIngestionPipeline:
    """Pipeline for batch ingestion of all cases from Pinecone to Neo4j."""
    
    def __init__(self):
        """Initialize the batch ingestion pipeline."""
        # Initialize Pinecone
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        self.namespace = config.PINECONE_NAMESPACE

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        
        # Initialize the single-case ingestion pipeline
        self.ingestion_pipeline = GraphIngestionPipeline()
        
        logger.info("BatchIngestionPipeline initialized")
    
    def get_all_case_extracts_from_pinecone(self, limit: int = None) -> Set[str]:
        """
        Query Pinecone iteratively to find unique case_extract_name values.
        Queries in batches of 10 with metadata filtering for efficiency.
        Stops early if we've found enough cases (when limit is reached).

        Args:
            limit: If specified, stop querying once we have this many unique case extracts

        Returns:
            Set of unique case extract names
        """
        logger.info("Querying Pinecone for all unique case extracts (iterative mode)...")

        case_extracts = set()
        batch_size = 10
        offset = 0
        max_iterations = 100  # Safety limit to prevent infinite loops

        try:
            # Get index stats to understand the data
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            logger.info(f"Total vectors in Pinecone index: {total_vectors}")

            if limit:
                logger.info(f"Will stop querying once {limit} unique case extracts are found")

            # Iteratively query in batches of 10
            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Iteration {iteration}: Querying batch starting at offset {offset}...")

                # Query with offset to get next batch
                # Note: Pinecone query doesn't support offset directly, so we use top_k with offset simulation
                results = self.index.query(
                    vector=[0] * 1024,  # Dummy vector (required by API)
                    top_k=batch_size + offset,  # Get more results to simulate offset
                    namespace=self.namespace,
                    include_metadata=True
                )

                matches = results.get('matches', [])

                # If we got fewer results than expected, we've reached the end
                if len(matches) <= offset:
                    logger.info(f"Reached end of results at iteration {iteration}")
                    break

                # Extract only the new batch (skip the ones we already processed)
                batch_matches = matches[offset:offset + batch_size]

                if not batch_matches:
                    logger.info(f"No more results at iteration {iteration}")
                    break

                # Extract unique case_extract_name values from this batch with metadata filtering
                batch_case_extracts = set()
                for match in batch_matches:
                    metadata = match.get('metadata', {})
                    case_extract_name = metadata.get('case_extract_name')

                    # Metadata filter: ensure case_extract_name exists and is not empty
                    if case_extract_name and isinstance(case_extract_name, str) and case_extract_name.strip():
                        batch_case_extracts.add(case_extract_name)
                        logger.debug(f"  Found case extract: {case_extract_name}")

                logger.info(f"Iteration {iteration}: Found {len(batch_case_extracts)} unique case extracts in this batch (total: {len(case_extracts) + len(batch_case_extracts)})")
                case_extracts.update(batch_case_extracts)

                # Early stopping: if we've found enough cases and limit is set, stop querying
                if limit and len(case_extracts) >= limit:
                    logger.info(f"Found {len(case_extracts)} case extracts (reached limit of {limit}). Stopping early.")
                    break

                # Move to next batch
                offset += batch_size

            logger.info(f"Total unique case extracts found: {len(case_extracts)}")

            # Log all found case extracts
            for case_extract in sorted(case_extracts):
                logger.info(f"  - {case_extract}")

            return case_extracts

        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return case_extracts  # Return what we found so far
    
    def get_loaded_cases_from_neo4j(self) -> Set[str]:
        """
        Query Neo4j to find all case numbers that have already been loaded.

        Returns:
            Set of case numbers (e.g., "2024_SGHCF_23")
        """
        logger.info("Querying Neo4j for already loaded cases...")

        loaded_cases = set()

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Case)
                    RETURN c.case_number as case_number
                """)

                for record in result:
                    case_number = record.get('case_number')
                    if case_number:
                        loaded_cases.add(case_number)

            logger.info(f"Found {len(loaded_cases)} cases already loaded in Neo4j")

            return loaded_cases

        except Exception as e:
            logger.error(f"Error querying Neo4j: {e}")
            return set()
    
    def case_extract_to_case_number(self, case_extract_name: str) -> str:
        """
        Convert case_extract_name to case_number format.
        
        Args:
            case_extract_name: e.g., "case_extract_2024_23"
            
        Returns:
            Case number in format "2024_SGHCF_23"
        """
        parts = case_extract_name.split("_")
        if len(parts) >= 3:
            year = parts[2]
            case_num = parts[3] if len(parts) > 3 else "1"
            return f"{year}_SGHCF_{case_num}"
        return ""
    
    def run_batch_ingestion(self, skip_loaded: bool = True, dry_run: bool = False, limit: int = None) -> None:
        """
        Run the batch ingestion process.
        Queries cases one at a time for better performance with small batches.

        Args:
            skip_loaded: If True, skip cases already loaded in Neo4j
            dry_run: If True, only show what would be ingested without actually ingesting
            limit: If specified, limit the number of cases to ingest
        """
        logger.info("=" * 80)
        logger.info("Starting batch ingestion process")
        logger.info("=" * 80)

        # Calculate how many cases we need to query for
        # If limit is set, we query for more than limit to account for already-loaded cases
        query_limit = None
        if limit and limit > 0:
            # Query for 2x the limit to account for already-loaded cases
            query_limit = limit * 2
            logger.info(f"Limit set to {limit}. Will query for up to {query_limit} case extracts.")

        # Get case extracts from Pinecone (iterative, with early stopping)
        all_case_extracts = self.get_all_case_extracts_from_pinecone(limit=query_limit)

        if not all_case_extracts:
            logger.warning("No case extracts found in Pinecone. Exiting.")
            return

        # Get already loaded cases from Neo4j (only if skip_loaded is True)
        loaded_cases = set()
        if skip_loaded:
            logger.info("Checking Neo4j for already loaded cases...")
            loaded_cases = self.get_loaded_cases_from_neo4j()

        # Determine which cases to ingest
        cases_to_ingest = []
        for case_extract in sorted(all_case_extracts):
            case_number = self.case_extract_to_case_number(case_extract)
            if case_number and (not skip_loaded or case_number not in loaded_cases):
                cases_to_ingest.append(case_extract)

        # Apply limit if specified
        if limit and limit > 0:
            cases_to_ingest = cases_to_ingest[:limit]

        logger.info("=" * 80)
        logger.info(f"Total case extracts found: {len(all_case_extracts)}")
        logger.info(f"Already loaded in Neo4j: {len(loaded_cases)}")
        if limit:
            logger.info(f"Limit applied: {limit}")
        logger.info(f"Cases to ingest: {len(cases_to_ingest)}")
        logger.info("=" * 80)

        if not cases_to_ingest:
            logger.info("No new cases to ingest. All cases are already loaded!")
            return

        # Show what will be ingested (only if limit is small)
        if len(cases_to_ingest) <= 10:
            logger.info("\nCases to be ingested:")
            for case_extract in cases_to_ingest:
                case_number = self.case_extract_to_case_number(case_extract)
                logger.info(f"  - {case_extract} → {case_number}")
        else:
            logger.info(f"\nCases to be ingested (showing first 5 of {len(cases_to_ingest)}):")
            for case_extract in cases_to_ingest[:5]:
                case_number = self.case_extract_to_case_number(case_extract)
                logger.info(f"  - {case_extract} → {case_number}")
            logger.info(f"  ... and {len(cases_to_ingest) - 5} more")

        if dry_run:
            logger.info("\n[DRY RUN] Skipping actual ingestion")
            return

        # Ingest each case
        logger.info("\n" + "=" * 80)
        logger.info("Starting ingestion...")
        logger.info("=" * 80 + "\n")

        success_count = 0
        failure_count = 0

        for idx, case_extract in enumerate(cases_to_ingest, 1):
            case_number = self.case_extract_to_case_number(case_extract)
            logger.info(f"\n[{idx}/{len(cases_to_ingest)}] Ingesting {case_extract} → {case_number}")
            logger.info("-" * 80)

            try:
                self.ingestion_pipeline.ingest_case(case_extract)
                success_count += 1
                logger.info(f"✅ Successfully ingested {case_extract}")
            except Exception as e:
                failure_count += 1
                logger.error(f"❌ Failed to ingest {case_extract}: {e}")
                # Continue with next case even if this one fails
                continue

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("BATCH INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total cases processed: {len(cases_to_ingest)}")
        logger.info(f"✅ Successful: {success_count}")
        logger.info(f"❌ Failed: {failure_count}")
        logger.info("=" * 80)
    
    def close(self):
        """Clean up resources."""
        self.driver.close()
        self.ingestion_pipeline.close()
        logger.info("Closed all connections")


def main():
    """Main function for running the batch ingestion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch ingest all cases from Pinecone into Neo4j"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of all cases, even if already loaded"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without actually ingesting"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of cases to ingest (useful for testing)"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = BatchIngestionPipeline()

    try:
        # Run batch ingestion
        pipeline.run_batch_ingestion(
            skip_loaded=not args.force,
            dry_run=args.dry_run,
            limit=args.limit
        )
    finally:
        # Clean up
        pipeline.close()


if __name__ == "__main__":
    main()

