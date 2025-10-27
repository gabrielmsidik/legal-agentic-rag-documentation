"""
Unit tests for FastAPI application.
Tests use actual credentials but perform read-only operations.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


class TestAPI:
    """Test cases for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Fixture to get TestClient instance."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "environment" in data
    
    def test_query_endpoint_with_valid_request(self, client):
        """Test the query endpoint with a valid request."""
        request_data = {
            "query": "What are cases about custody of children?"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
    
    def test_query_endpoint_with_missing_query(self, client):
        """Test the query endpoint with missing query field."""
        request_data = {}
        response = client.post("/query", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_response_structure(self, client):
        """Test that query endpoint returns correct response structure."""
        request_data = {
            "query": "matrimonial assets division"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "answer" in data
        assert "sources" in data

        # Check sources structure if any
        if data["sources"]:
            source = data["sources"][0]
            assert "type" in source
            assert source["type"] in ["vector", "graph"]
            assert "data" in source
            assert "metadata" in source

    def test_query_custody_disputes(self, client):
        """Test query about child custody disputes."""
        request_data = {
            "query": "Find cases involving child custody disputes"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_legal_principles(self, client):
        """Test query about legal principles in specific cases."""
        request_data = {
            "query": "What legal principles are mentioned in recent cases?"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "sources" in data

    def test_query_maintenance_support(self, client):
        """Test query about maintenance and support cases."""
        request_data = {
            "query": "What are cases about spousal maintenance and support?"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_family_law_framework(self, client):
        """Test query about family law frameworks."""
        request_data = {
            "query": "What are the key frameworks in family law?"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_query_judicial_reasoning(self, client):
        """Test query about judicial reasoning and arguments."""
        request_data = {
            "query": "Show me arguments made by judges in recent cases"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_case_parties(self, client):
        """Test query about parties involved in cases."""
        request_data = {
            "query": "Who are the key parties and judges in family law cases?"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_ancillary_matters(self, client):
        """Test query about ancillary matters in family law."""
        request_data = {
            "query": "What are ancillary matters in family law proceedings?"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_query_asset_valuation(self, client):
        """Test query about asset valuation in matrimonial cases."""
        request_data = {
            "query": "How are matrimonial assets valued and divided?"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) > 0

    def test_query_short_marriages(self, client):
        """Test query about short marriage cases."""
        request_data = {
            "query": "What are the considerations for short marriages?"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_empty_string(self, client):
        """Test query endpoint with empty query string."""
        request_data = {
            "query": ""
        }
        response = client.post("/query", json=request_data)
        # Should still return 200 but with minimal results
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_very_long_query(self, client):
        """Test query endpoint with very long query string."""
        long_query = "What are the legal principles and frameworks for " * 20
        request_data = {
            "query": long_query
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_special_characters(self, client):
        """Test query with special characters."""
        request_data = {
            "query": "Cases with @ # $ % ^ & * symbols in matrimonial law"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_multiple_topics(self, client):
        """Test query combining multiple legal topics."""
        request_data = {
            "query": "custody, maintenance, and asset division in family law"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_joint_accounts_with_third_parties(self, client):
        """Test query about joint accounts with siblings/parents in matrimonial asset division.

        This test validates the specific query:
        "How are joint accounts with other parties (siblings/parents) dealt with in division of matrimonial assets?"

        It checks:
        1. Response structure is valid
        2. Answer is generated
        3. Sources are retrieved (both vector and graph-based)
        4. Detailed breakdown of source types for debugging
        """
        request_data = {
            "query": "How are joint accounts with other parties (siblings/parents) dealt with in division of matrimonial assets?"
        }
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Basic assertions
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0, "Answer should not be empty"

        assert "sources" in data
        assert isinstance(data["sources"], list)

        # Detailed source analysis
        sources = data["sources"]
        vector_sources = [s for s in sources if s.get("type") == "vector"]
        graph_sources = [s for s in sources if s.get("type") == "graph"]

        # Log source breakdown for debugging
        print(f"\n{'='*80}")
        print(f"Query: {request_data['query']}")
        print(f"{'='*80}")
        print(f"Total sources: {len(sources)}")
        print(f"Vector sources: {len(vector_sources)}")
        print(f"Graph sources: {len(graph_sources)}")

        if vector_sources:
            print(f"\nVector Sources:")
            for i, source in enumerate(vector_sources, 1):
                print(f"  {i}. {source.get('metadata', {}).get('case_extract_name', 'Unknown')}")
                print(f"     Score: {source.get('metadata', {}).get('score', 'N/A')}")

        if graph_sources:
            print(f"\nGraph Sources:")
            for i, source in enumerate(graph_sources, 1):
                data = source.get('data', {})
                if isinstance(data, dict):
                    # Extract a meaningful summary from the dict
                    summary = data.get('c.summary', data.get('a.summary', str(data)[:100]))
                else:
                    summary = str(data)[:100]
                print(f"  {i}. {summary}...")
        else:
            print(f"\n⚠️  WARNING: No graph-based sources found!")
            print(f"This may indicate:")
            print(f"  - Graph search didn't find relevant entities")
            print(f"  - Query keywords don't match graph node properties")
            print(f"  - Graph search strategy needs optimization")

        print(f"{'='*80}\n")

        # At least some sources should be found
        assert len(sources) > 0, "Should find at least some sources (vector or graph)"

    def test_query_joint_accounts_with_parents(self, client):
        """Test query about joint accounts with parents in matrimonial asset division."""
        request_data = {
            "query": "How are joint accounts with parents dealt with in splitting of matrimonial assets?"
        }
        response = client.post("/query", json=request_data)

        print(f"\n{'='*80}")
        print(f"TEST: Joint Accounts with Parents Query")
        print(f"{'='*80}")
        print(f"Query: {request_data['query']}")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "workflow_nodes" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["workflow_nodes"], list)

        print(f"\nAnswer: {data['answer'][:200]}...")
        print(f"\nTotal Sources: {len(data['sources'])}")
        print(f"Total Workflow Nodes: {len(data['workflow_nodes'])}")

        # Verify workflow_nodes is populated
        assert len(data["workflow_nodes"]) > 0, "workflow_nodes should not be empty"

        # Verify workflow node structure
        for node in data["workflow_nodes"]:
            assert "node_name" in node, "Workflow node should have 'node_name' field"
            assert "node_type" in node, "Workflow node should have 'node_type' field"
            assert "results_count" in node, "Workflow node should have 'results_count' field"
            assert "summary" in node, "Workflow node should have 'summary' field"
            assert "execution_order" in node, "Workflow node should have 'execution_order' field"

            print(f"\n[{node['execution_order']}] {node['node_name']} ({node['node_type']})")
            print(f"  Results: {node['results_count']}")
            print(f"  Summary: {node['summary']}")

            # Verify node types
            assert node['node_type'] in ['tool', 'decision', 'synthesis'], f"Invalid node_type: {node['node_type']}"

            # Verify execution order is sequential
            assert node['execution_order'] == len([n for n in data["workflow_nodes"] if n['execution_order'] <= node['execution_order']]), \
                "Execution order should be sequential"

        print(f"\n{'='*80}\n")

        # At least some sources should be found
        assert len(data["sources"]) > 0, "Should find at least some sources"

