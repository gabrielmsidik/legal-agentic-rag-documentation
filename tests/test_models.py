"""
Unit tests for Pydantic models and schemas.
"""

import pytest
from pydantic import ValidationError
from src.models.schemas import (
    QueryRequest,
    QueryResponse,
    Source,
    VectorSearchInput,
    GraphSearchInput,
    CaseNode,
    PersonNode,
    StatuteNode,
    LegalPrincipleNode,
    ArgumentNode,
)


class TestQueryRequest:
    """Test cases for QueryRequest model."""
    
    def test_valid_query_request(self):
        """Test creating a valid QueryRequest."""
        request = QueryRequest(query="What are the cases about custody?")
        assert request.query == "What are the cases about custody?"
    
    def test_query_request_requires_query(self):
        """Test that QueryRequest requires a query field."""
        with pytest.raises(ValidationError):
            QueryRequest()


class TestSource:
    """Test cases for Source model."""
    
    def test_vector_source(self):
        """Test creating a vector source."""
        source = Source(
            type="vector",
            chunk_id="case_extract_2024_1_section_1",
            data="This is the text content",
            metadata={"case_number": "2024_SGHCF_1"}
        )
        assert source.type == "vector"
        assert source.chunk_id == "case_extract_2024_1_section_1"
        assert source.data == "This is the text content"
    
    def test_graph_source(self):
        """Test creating a graph source."""
        source = Source(
            type="graph",
            chunk_id=None,
            data={"argument": "summary", "outcome": "Accepted"},
            metadata={"generated_cypher": "MATCH (a:Argument)..."}
        )
        assert source.type == "graph"
        assert source.chunk_id is None
        assert isinstance(source.data, dict)


class TestQueryResponse:
    """Test cases for QueryResponse model."""
    
    def test_valid_query_response(self):
        """Test creating a valid QueryResponse."""
        sources = [
            Source(type="vector", chunk_id="chunk_1", data="text", metadata={})
        ]
        response = QueryResponse(
            answer="This is the answer",
            sources=sources
        )
        assert response.answer == "This is the answer"
        assert len(response.sources) == 1
    
    def test_query_response_with_empty_sources(self):
        """Test QueryResponse with empty sources list."""
        response = QueryResponse(answer="Answer", sources=[])
        assert response.answer == "Answer"
        assert len(response.sources) == 0


class TestVectorSearchInput:
    """Test cases for VectorSearchInput model."""
    
    def test_semantic_search_input(self):
        """Test VectorSearchInput for semantic search."""
        input_data = VectorSearchInput(query="custody cases", top_k=10)
        assert input_data.query == "custody cases"
        assert input_data.fetch_ids is None
        assert input_data.top_k == 10
    
    def test_fetch_by_ids_input(self):
        """Test VectorSearchInput for fetching by IDs."""
        input_data = VectorSearchInput(fetch_ids=["chunk_1", "chunk_2"])
        assert input_data.query is None
        assert len(input_data.fetch_ids) == 2


class TestGraphSearchInput:
    """Test cases for GraphSearchInput model."""
    
    def test_valid_graph_search_input(self):
        """Test creating a valid GraphSearchInput."""
        input_data = GraphSearchInput(query="Find all cases about custody")
        assert input_data.query == "Find all cases about custody"


class TestCaseNode:
    """Test cases for CaseNode model."""
    
    def test_valid_case_node(self):
        """Test creating a valid CaseNode."""
        case = CaseNode(
            case_number="2024_SGHCF_1",
            name="ABC v XYZ",
            date="2024-01-15",
            year=2024,
            summary="Case about custody",
            outcome="Appeal allowed"
        )
        assert case.case_number == "2024_SGHCF_1"
        assert case.year == 2024


class TestPersonNode:
    """Test cases for PersonNode model."""
    
    def test_judge_person_node(self):
        """Test creating a Person node for a judge."""
        person = PersonNode(name="Tan Lee Meng", role="Judge")
        assert person.name == "Tan Lee Meng"
        assert person.role == "Judge"
    
    def test_party_person_node(self):
        """Test creating a Person node for a party."""
        person = PersonNode(
            name="XDJ-Appellant-2024_SGHCF_1",
            role="Appellant"
        )
        assert "Appellant" in person.name
        assert person.role == "Appellant"


class TestStatuteNode:
    """Test cases for StatuteNode model."""
    
    def test_valid_statute_node(self):
        """Test creating a valid StatuteNode."""
        statute = StatuteNode(
            name="Women's Charter (Cap 353)",
            section="Section 112"
        )
        assert statute.name == "Women's Charter (Cap 353)"
        assert statute.section == "Section 112"


class TestLegalPrincipleNode:
    """Test cases for LegalPrincipleNode model."""
    
    def test_valid_legal_principle_node(self):
        """Test creating a valid LegalPrincipleNode."""
        principle = LegalPrincipleNode(name="Division of Matrimonial Assets")
        assert principle.name == "Division of Matrimonial Assets"


class TestArgumentNode:
    """Test cases for ArgumentNode model."""
    
    def test_valid_argument_node(self):
        """Test creating a valid ArgumentNode."""
        argument = ArgumentNode(
            summary="Party claims equal division",
            judge_reasoning="Court finds unequal contribution",
            outcome="Rejected",
            section="Section 7",
            vector_chunk_ids=["case_extract_2024_1_7"]
        )
        assert argument.outcome == "Rejected"
        assert len(argument.vector_chunk_ids) == 1
    
    def test_argument_with_multiple_chunks(self):
        """Test ArgumentNode with multiple vector chunk IDs."""
        argument = ArgumentNode(
            summary="Summary",
            judge_reasoning="Reasoning",
            outcome="Accepted",
            section="Section 5",
            vector_chunk_ids=["chunk_1", "chunk_2", "chunk_3"]
        )
        assert len(argument.vector_chunk_ids) == 3

