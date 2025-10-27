"""
Debug script to trace the agent flow and understand why vector search might not be running.
Also tests the evaluate_results_node logic.
"""

import logging
from src.api.agent import LegalRAGAgent
from src.models.schemas import GraphState

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test query
query = "How are joint accounts with other parties (siblings/parents) dealt with in division of matrimonial assets?"

print("="*80)
print("DEBUGGING AGENT FLOW")
print("="*80)
print(f"\nQuery: {query}\n")

agent = LegalRAGAgent()

# Initialize state
initial_state: GraphState = {
    "original_query": query,
    "retrieved_context": [],
    "intermediate_steps": [],
    "iteration_count": 0,
    "plan": "",
    "needs_more_info": False,
}

print("Step 1: Initial Plan Decision")
print("-"*80)
state = agent.plan_step(initial_state)
print(f"Plan: {state['plan']}")
print(f"Iteration count: {state['iteration_count']}\n")

# Execute first search based on plan
if state['plan'] == 'vector_search':
    print("Step 2: Executing Vector Search")
    print("-"*80)
    state = agent.vector_search_node(state)
    print(f"Retrieved {len(state['retrieved_context'])} sources")
    print(f"Intermediate steps: {state['intermediate_steps']}\n")
elif state['plan'] == 'graph_search':
    print("Step 2: Executing Graph Search")
    print("-"*80)
    state = agent.graph_search_node(state)
    print(f"Retrieved {len(state['retrieved_context'])} sources")
    print(f"Intermediate steps: {state['intermediate_steps']}\n")

# Evaluate results
print("Step 3: Evaluating Results")
print("-"*80)
state = agent.evaluate_results_node(state)
print(f"Needs more info: {state['needs_more_info']}")
print(f"Retrieved context count: {len(state['retrieved_context'])}\n")

# If needs more info, do second search
if state['needs_more_info']:
    print("Step 4: Second Search Decision")
    print("-"*80)
    state = agent.plan_step(state)
    print(f"Plan: {state['plan']}")
    print(f"Iteration count: {state['iteration_count']}\n")
    
    if state['plan'] == 'vector_search':
        print("Step 5: Executing Vector Search (2nd)")
        print("-"*80)
        state = agent.vector_search_node(state)
        print(f"Retrieved {len(state['retrieved_context'])} sources total")
        print(f"Intermediate steps: {state['intermediate_steps']}\n")
    elif state['plan'] == 'graph_search':
        print("Step 5: Executing Graph Search (2nd)")
        print("-"*80)
        state = agent.graph_search_node(state)
        print(f"Retrieved {len(state['retrieved_context'])} sources total")
        print(f"Intermediate steps: {state['intermediate_steps']}\n")
    
    # Evaluate again
    print("Step 6: Evaluating Results (2nd)")
    print("-"*80)
    state = agent.evaluate_results_node(state)
    print(f"Needs more info: {state['needs_more_info']}")
    print(f"Retrieved context count: {len(state['retrieved_context'])}\n")

# Final synthesis
print("Step 7: Synthesizing Answer")
print("-"*80)
state = agent.synthesize_answer_node(state)
print(f"Answer length: {len(state['answer'])} characters")
print(f"Answer preview: {state['answer'][:200]}...\n")

print("="*80)
print("DEBUG COMPLETE")
print("="*80)

