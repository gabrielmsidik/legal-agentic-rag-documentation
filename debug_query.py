"""
Debug script to test query processing step by step.
"""

import logging
from src.api.agent import get_agent
from src.tools import get_vector_search_tool, get_graph_search_tool

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_vector_search():
    """Test vector search directly."""
    print("\n" + "="*80)
    print("TESTING VECTOR SEARCH")
    print("="*80)
    
    tool = get_vector_search_tool()
    query = "What are cases about custody of children?"
    
    print(f"\nQuery: {query}")
    results = tool.semantic_search(query, top_k=5)
    
    print(f"\nResults count: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.chunk_id}")
        print(f"   Type: {result.type}")
        print(f"   Score: {result.metadata.get('score', 'N/A')}")
        print(f"   Data preview: {result.data[:100]}...")

def test_graph_search():
    """Test graph search directly."""
    print("\n" + "="*80)
    print("TESTING GRAPH SEARCH")
    print("="*80)
    
    tool = get_graph_search_tool()
    query = "What are cases about custody of children?"
    
    print(f"\nQuery: {query}")
    results = tool.search(query)
    
    print(f"\nResults count: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.chunk_id}")
        print(f"   Type: {result.type}")
        print(f"   Data: {result.data}")

def test_agent():
    """Test the full agent."""
    print("\n" + "="*80)
    print("TESTING FULL AGENT")
    print("="*80)
    
    agent = get_agent()
    query = "What are cases about custody of children?"
    
    print(f"\nQuery: {query}")
    result = agent.run(query)
    
    print(f"\nAnswer length: {len(result['answer'])} chars")
    print(f"Sources count: {len(result['sources'])}")
    print(f"\nAnswer preview:\n{result['answer'][:200]}...")
    
    if result['sources']:
        print(f"\nFirst source:")
        source = result['sources'][0]
        print(f"  Type: {source.type}")
        print(f"  Chunk ID: {source.chunk_id}")
        print(f"  Data preview: {source.data[:100]}...")

if __name__ == "__main__":
    try:
        test_vector_search()
        test_graph_search()
        test_agent()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

