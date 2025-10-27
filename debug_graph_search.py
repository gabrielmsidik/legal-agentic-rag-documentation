"""
Debug script to investigate why graph search returns no results for the joint accounts query.
"""

import logging
from src.tools.graph_search import GraphSearchTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

query = "How are joint accounts with other parties (siblings/parents) dealt with in division of matrimonial assets?"

print("="*80)
print("DEBUGGING GRAPH SEARCH")
print("="*80)
print(f"\nQuery: {query}\n")

graph_tool = GraphSearchTool()

# Step 1: Generate Cypher query
print("Step 1: Generating Cypher query from natural language...")
print("-"*80)
cypher_query = graph_tool._generate_cypher_query(query)
print(f"Generated Cypher:\n{cypher_query}\n")

# Step 2: Execute the Cypher query
print("Step 2: Executing Cypher query...")
print("-"*80)
results = graph_tool.execute_cypher(cypher_query)
print(f"Results count: {len(results)}")
if results:
    print(f"Results:\n{results}\n")
else:
    print("No results found!\n")

# Step 3: Try some alternative queries to understand the data
print("Step 3: Exploring what data exists in the graph...")
print("-"*80)

# Check what legal principles exist
print("\nAvailable Legal Principles:")
lp_query = "MATCH (lp:LegalPrinciple) RETURN DISTINCT lp.name LIMIT 20"
lp_results = graph_tool.execute_cypher(lp_query)
for result in lp_results:
    print(f"  - {result.get('lp.name', 'Unknown')}")

# Check what cases exist
print("\nSample Cases:")
case_query = "MATCH (c:Case) RETURN c.case_number, c.name, c.summary LIMIT 5"
case_results = graph_tool.execute_cypher(case_query)
for result in case_results:
    print(f"  - {result.get('c.case_number', 'Unknown')}: {result.get('c.name', 'Unknown')}")

# Check what arguments exist
print("\nSample Arguments:")
arg_query = "MATCH (a:Argument) RETURN a.summary LIMIT 5"
arg_results = graph_tool.execute_cypher(arg_query)
for result in arg_results:
    summary = result.get('a.summary', 'Unknown')
    if len(summary) > 80:
        summary = summary[:80] + "..."
    print(f"  - {summary}")

graph_tool.close()

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)

