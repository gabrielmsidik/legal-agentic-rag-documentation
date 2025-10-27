"""
Unit tests for graph ingestion pipeline.
Tests use actual credentials. Includes small isolated create+delete tests.
"""

import pytest
from src.ingestion.graph_ingestion import GraphIngestionPipeline
from neo4j import GraphDatabase
from src.config import config


class TestGraphIngestionPipeline:
    """Test cases for GraphIngestionPipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Fixture to get GraphIngestionPipeline instance."""
        return GraphIngestionPipeline()
    
    @pytest.fixture
    def neo4j_driver(self):
        """Fixture to get Neo4j driver for cleanup."""
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        yield driver
        driver.close()
    
    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.pc is not None
        assert pipeline.index is not None
        assert pipeline.driver is not None
        assert pipeline.llm is not None
    
    def test_extract_first_pass_returns_dict(self, pipeline):
        """Test that extract_first_pass returns a dictionary."""
        sample_text = """
        Case: ABC v XYZ [2024] SGHCF 1
        Date: 15 January 2024
        Judge: Justice Tan Lee Meng
        
        This case concerns the division of matrimonial assets.
        The appellant sought a larger share of the assets.
        The court dismissed the appeal.
        """
        
        result = pipeline.extract_first_pass(sample_text, "2024_SGHCF_1")
        assert isinstance(result, dict)
        assert "case" in result or "persons" in result or "legal_principles" in result
    
    def test_extract_second_pass_returns_dict(self, pipeline):
        """Test that extract_second_pass returns a dictionary."""
        sample_text = """
        The appellant argued for equal division under Section 112 of the Women's Charter.
        The court found that contributions were unequal and rejected this argument.
        """
        
        result = pipeline.extract_second_pass(sample_text, "2024_SGHCF_1")
        assert isinstance(result, dict)
    
    def test_ingest_to_neo4j_creates_nodes(self, pipeline, neo4j_driver):
        """Test that ingest_to_neo4j creates nodes (with cleanup)."""
        # Create test data
        test_case_number = "TEST_2024_SGHCF_999"
        extraction_data = {
            "case": {
                "case_number": test_case_number,
                "name": "Test Case",
                "date": "2024-01-01",
                "year": 2024,
                "summary": "Test summary",
                "outcome": "Test outcome"
            },
            "persons": [
                {"name": "Test Judge", "role": "Judge"}
            ],
            "legal_principles": [
                {"name": "Test Principle"}
            ],
            "arguments": []
        }
        
        try:
            # Ingest the test data
            pipeline.ingest_to_neo4j(extraction_data, test_case_number)
            
            # Verify the case was created
            with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run(
                    "MATCH (c:Case {case_number: $case_number}) RETURN c",
                    case_number=test_case_number
                )
                record = result.single()
                assert record is not None
                assert record["c"]["name"] == "Test Case"
        
        finally:
            # Cleanup: Delete the test case and related nodes
            with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                session.run("""
                    MATCH (c:Case {case_number: $case_number})
                    OPTIONAL MATCH (c)-[r]-()
                    DELETE r, c
                """, case_number=test_case_number)
                
                # Clean up test person
                session.run("""
                    MATCH (p:Person {name: 'Test Judge'})
                    OPTIONAL MATCH (p)-[r]-()
                    DELETE r, p
                """)
                
                # Clean up test principle
                session.run("""
                    MATCH (lp:LegalPrinciple {name: 'Test Principle'})
                    OPTIONAL MATCH (lp)-[r]-()
                    DELETE r, lp
                """)
    
    def test_ingest_to_neo4j_creates_relationships(self, pipeline, neo4j_driver):
        """Test that ingest_to_neo4j creates relationships (with cleanup)."""
        test_case_number = "TEST_2024_SGHCF_998"
        extraction_data = {
            "case": {
                "case_number": test_case_number,
                "name": "Test Case 2",
                "date": "2024-01-01",
                "year": 2024,
                "summary": "Test",
                "outcome": "Test"
            },
            "persons": [
                {"name": "Test Judge 2", "role": "Judge"}
            ],
            "legal_principles": [
                {"name": "Test Principle 2"}
            ],
            "arguments": []
        }
        
        try:
            # Ingest the test data
            pipeline.ingest_to_neo4j(extraction_data, test_case_number)
            
            # Verify relationship was created
            with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                result = session.run("""
                    MATCH (p:Person {name: 'Test Judge 2'})-[:PRESIDED_OVER]->(c:Case {case_number: $case_number})
                    RETURN p, c
                """, case_number=test_case_number)
                record = result.single()
                assert record is not None
        
        finally:
            # Cleanup
            with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                session.run("""
                    MATCH (c:Case {case_number: $case_number})
                    OPTIONAL MATCH (c)-[r]-()
                    DELETE r, c
                """, case_number=test_case_number)
                
                session.run("""
                    MATCH (p:Person {name: 'Test Judge 2'})
                    OPTIONAL MATCH (p)-[r]-()
                    DELETE r, p
                """)
                
                session.run("""
                    MATCH (lp:LegalPrinciple {name: 'Test Principle 2'})
                    OPTIONAL MATCH (lp)-[r]-()
                    DELETE r, lp
                """)

