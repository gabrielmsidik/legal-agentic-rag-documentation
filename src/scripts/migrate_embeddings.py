"""
Migration script to regenerate embeddings from 1024 dimensions (Pinecone) to 1536 dimensions (OpenAI).

This script:
1. Connects to Neo4j
2. Finds all nodes with embeddings
3. Regenerates embeddings using OpenAI's API
4. Updates the nodes with new embeddings

Usage:
    python scripts/migrate_embeddings.py
"""

import logging
from neo4j import GraphDatabase
from src.config import config
from src.tools.embedding_utils import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingMigration:
    """Migrate embeddings from Pinecone (1024 dims) to OpenAI (1536 dims)."""
    
    def __init__(self):
        """Initialize Neo4j driver and embedding generator."""
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        self.database = config.NEO4J_DATABASE
        self.embedding_gen = EmbeddingGenerator()
        
    def close(self):
        """Close Neo4j driver."""
        self.driver.close()
    
    def migrate_case_embeddings(self, batch_size: int = 10):
        """Regenerate embeddings for Case nodes."""
        logger.info("Starting Case node embedding migration...")
        
        with self.driver.session(database=self.database) as session:
            # Get all Case nodes with embeddings
            result = session.run("""
                MATCH (c:Case)
                WHERE c.summary IS NOT NULL OR c.outcome IS NOT NULL
                RETURN c.case_number as case_number, c.summary as summary, c.outcome as outcome, c.summary_embedding as summary_embedding 
            """)
            
            cases = [record for record in result]
            logger.info(f"Found {len(cases)} Case nodes to migrate")
            
            for i, case in enumerate(cases):
                case_number = case['case_number']
                keys = case.keys()
                logger.info(f"Keys of the case: {keys}")
                summary_embedding = case['summary_embedding']

                if len(summary_embedding) == 1536:
                    logger.info(f"Case {case_number} already has 1536-dimensional embedding for summary. Skipping...")
                    continue

                summary = case['summary']
                outcome = case['outcome']
                
                # Generate new embeddings
                summary_embedding = self.embedding_gen.generate_embedding(summary) if summary else None
                outcome_embedding = self.embedding_gen.generate_embedding(outcome) if outcome else None
                
                # Update the node
                session.run("""
                    MATCH (c:Case {case_number: $case_number})
                    SET c.summary_embedding = $summary_embedding,
                        c.outcome_embedding = $outcome_embedding
                """, 
                    case_number=case_number,
                    summary_embedding=summary_embedding,
                    outcome_embedding=outcome_embedding
                )
                
                if (i + 1) % batch_size == 0:
                    logger.info(f"Migrated {i + 1}/{len(cases)} Case nodes")
        
        logger.info(f"Completed Case node migration: {len(cases)} nodes updated")
    
    def migrate_argument_embeddings(self, batch_size: int = 10):
        """Regenerate embeddings for Argument nodes."""
        logger.info("Starting Argument node embedding migration...")
        
        with self.driver.session(database=self.database) as session:
            # Get all Argument nodes with embeddings
            result = session.run("""
                MATCH (a:Argument)
                WHERE a.summary IS NOT NULL OR a.judge_reasoning IS NOT NULL
                RETURN elementId(a) as arg_id, a.summary as summary, a.judge_reasoning as judge_reasoning, a.summary_embedding as summary_embedding
            """)
            
            arguments = [record for record in result]
            logger.info(f"Found {len(arguments)} Argument nodes to migrate")
            
            for i, arg in enumerate(arguments):
                arg_id = arg['arg_id']
                summary = arg['summary']
                judge_reasoning = arg['judge_reasoning']
                summary_embedding = arg['summary_embedding']

                if summary_embedding and len(summary_embedding) == 1536:
                    logger.info(f"Argument {arg_id} already has 1536-dimensional embedding for summary. Skipping...")
                    continue
                
                # Generate new embeddings
                summary_embedding = self.embedding_gen.generate_embedding(summary) if summary else None
                judge_reasoning_embedding = self.embedding_gen.generate_embedding(judge_reasoning) if judge_reasoning else None
                
                # Update the node
                session.run("""
                    MATCH (a:Argument) WHERE elementId(a) = $arg_id
                    SET a.summary_embedding = $summary_embedding,
                        a.judge_reasoning_embedding = $judge_reasoning_embedding
                """,
                    arg_id=arg_id,
                    summary_embedding=summary_embedding,
                    judge_reasoning_embedding=judge_reasoning_embedding
                )
                
                if (i + 1) % batch_size == 0:
                    logger.info(f"Migrated {i + 1}/{len(arguments)} Argument nodes")
        
        logger.info(f"Completed Argument node migration: {len(arguments)} nodes updated")
    
    def migrate_statute_embeddings(self, batch_size: int = 10):
        """Regenerate embeddings for Statute nodes."""
        logger.info("Starting Statute node embedding migration...")
        
        with self.driver.session(database=self.database) as session:
            # Get all Statute nodes with embeddings
            result = session.run("""
                MATCH (s:Statute)
                WHERE s.summary IS NOT NULL
                RETURN s.name as name, s.section as section, s.summary as summary, s.summary_embedding as summary_embedding
            """)
            
            statutes = [record for record in result]
            logger.info(f"Found {len(statutes)} Statute nodes to migrate")
            
            for i, statute in enumerate(statutes):
                name = statute['name']
                section = statute['section']
                summary = statute['summary']
                summary_embedding = statute['summary_embedding']

                if len(summary_embedding) == 1536:
                    logger.info(f"Statute {name} {section} already has 1536-dimensional embedding for summary. Skipping...")
                    continue

                # Generate new embedding
                summary_embedding = self.embedding_gen.generate_embedding(summary) if summary else None
                
                # Update the node
                session.run("""
                    MATCH (s:Statute {name: $name, section: $section})
                    SET s.summary_embedding = $summary_embedding
                """,
                    name=name,
                    section=section,
                    summary_embedding=summary_embedding
                )
                
                if (i + 1) % batch_size == 0:
                    logger.info(f"Migrated {i + 1}/{len(statutes)} Statute nodes")
        
        logger.info(f"Completed Statute node migration: {len(statutes)} nodes updated")
    
    def run_migration(self):
        """Run complete embedding migration."""
        logger.info("=" * 80)
        logger.info("EMBEDDING MIGRATION: Pinecone (1024 dims) → OpenAI (1536 dims)")
        logger.info("=" * 80)
        
        try:
            self.migrate_case_embeddings()
            self.migrate_argument_embeddings()
            self.migrate_statute_embeddings()
            
            logger.info("=" * 80)
            logger.info("✅ MIGRATION COMPLETE")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            self.close()


if __name__ == "__main__":
    migration = EmbeddingMigration()
    migration.run_migration()

