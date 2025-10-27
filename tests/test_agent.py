"""
Unit tests for LangGraph agent.
Tests use actual credentials but perform read-only operations.
"""

import pytest
from src.api.agent import LegalRAGAgent, get_agent
from src.models.schemas import GraphState


class TestLegalRAGAgent:
    """Test cases for LegalRAGAgent."""
    
    @pytest.fixture
    def agent(self):
        """Fixture to get LegalRAGAgent instance."""
        return get_agent()
    
    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent is not None
        assert agent.llm is not None
        assert agent.vector_tool is not None
        assert agent.graph_tool is not None
        assert agent.graph is not None
    
    def test_singleton_pattern(self):
        """Test that get_agent returns the same instance."""
        agent1 = get_agent()
        agent2 = get_agent()
        assert agent1 is agent2
    
    def test_plan_step_updates_state(self, agent):
        """Test that plan_step updates the state correctly."""
        state: GraphState = {
            "original_query": "What are cases about custody?",
            "intermediate_steps": [],
            "retrieved_context": [],
            "plan": None,
            "response": None,
            "needs_more_info": True,
            "iteration_count": 0,
        }
        
        new_state = agent.plan_step(state)
        assert new_state["plan"] is not None
        assert new_state["plan"] in ["vector_search", "graph_search", "synthesize"]
        assert new_state["iteration_count"] == 1
    
    def test_route_action_returns_valid_route(self, agent):
        """Test that route_action returns valid routes."""
        state_vector: GraphState = {
            "original_query": "test",
            "intermediate_steps": [],
            "retrieved_context": [],
            "plan": "vector_search",
            "response": None,
            "needs_more_info": True,
            "iteration_count": 0,
        }
        assert agent.route_action(state_vector) == "vector"
        
        state_graph: GraphState = {
            "original_query": "test",
            "intermediate_steps": [],
            "retrieved_context": [],
            "plan": "graph_search",
            "response": None,
            "needs_more_info": True,
            "iteration_count": 0,
        }
        assert agent.route_action(state_graph) == "graph"
        
        state_synth: GraphState = {
            "original_query": "test",
            "intermediate_steps": [],
            "retrieved_context": [],
            "plan": "synthesize",
            "response": None,
            "needs_more_info": True,
            "iteration_count": 0,
        }
        assert agent.route_action(state_synth) == "synthesize"
    
    def test_vector_search_node_updates_context(self, agent):
        """Test that vector_search_node updates retrieved_context."""
        state: GraphState = {
            "original_query": "custody cases",
            "intermediate_steps": [],
            "retrieved_context": [],
            "plan": "vector_search",
            "response": None,
            "needs_more_info": True,
            "iteration_count": 0,
        }
        
        new_state = agent.vector_search_node(state)
        assert isinstance(new_state["retrieved_context"], list)
        assert len(new_state["intermediate_steps"]) > 0
    
    def test_graph_search_node_updates_context(self, agent):
        """Test that graph_search_node updates retrieved_context."""
        state: GraphState = {
            "original_query": "Find cases about custody",
            "intermediate_steps": [],
            "retrieved_context": [],
            "plan": "graph_search",
            "response": None,
            "needs_more_info": True,
            "iteration_count": 0,
        }
        
        new_state = agent.graph_search_node(state)
        assert isinstance(new_state["retrieved_context"], list)
        assert len(new_state["intermediate_steps"]) > 0
    
    def test_evaluate_results_node_sets_needs_more_info(self, agent):
        """Test that evaluate_results_node sets needs_more_info flag."""
        state: GraphState = {
            "original_query": "test query",
            "intermediate_steps": [],
            "retrieved_context": [],
            "plan": None,
            "response": None,
            "needs_more_info": True,
            "iteration_count": 0,
        }
        
        new_state = agent.evaluate_results_node(state)
        assert "needs_more_info" in new_state
        assert isinstance(new_state["needs_more_info"], bool)
    
    def test_should_continue_logic(self, agent):
        """Test should_continue conditional logic."""
        state_continue: GraphState = {
            "original_query": "test",
            "intermediate_steps": [],
            "retrieved_context": [],
            "plan": None,
            "response": None,
            "needs_more_info": True,
            "iteration_count": 0,
        }
        assert agent.should_continue(state_continue) == "continue"
        
        state_synth: GraphState = {
            "original_query": "test",
            "intermediate_steps": [],
            "retrieved_context": [],
            "plan": None,
            "response": None,
            "needs_more_info": False,
            "iteration_count": 0,
        }
        assert agent.should_continue(state_synth) == "synthesize"
    
    def test_synthesize_answer_node_generates_response(self, agent):
        """Test that synthesize_answer_node generates a response."""
        state: GraphState = {
            "original_query": "What is custody?",
            "intermediate_steps": [],
            "retrieved_context": [
                {
                    "type": "vector",
                    "chunk_id": "test_chunk",
                    "data": "Custody refers to legal guardianship.",
                    "metadata": {}
                }
            ],
            "plan": None,
            "response": None,
            "needs_more_info": False,
            "iteration_count": 0,
        }
        
        new_state = agent.synthesize_answer_node(state)
        assert new_state["response"] is not None
        assert isinstance(new_state["response"], str)
        assert len(new_state["response"]) > 0
    
    def test_run_method_returns_result(self, agent):
        """Test that run method returns a valid result."""
        result = agent.run("What are cases about custody?")
        
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)

