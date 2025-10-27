"""
Simple script to test the Legal RAG API with sample queries.
"""

import requests
import json
import sys

API_URL = "http://localhost:8000/query"

def test_query(query: str):
    """Send a query to the API and print the response."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}\n")
    
    try:
        response = requests.post(
            API_URL,
            json={"query": query},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS\n")
            print(f"Answer:\n{result.get('answer', 'No answer provided')}\n")
            
            # Print sources if available
            sources = result.get('sources', [])
            if sources:
                print(f"\nSources ({len(sources)}):")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")
            
            # Print metadata if available
            metadata = result.get('metadata', {})
            if metadata:
                print(f"\nMetadata:")
                print(f"  - Tools used: {metadata.get('tools_used', [])}")
                print(f"  - Steps: {metadata.get('steps', 0)}")
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out (>120s)")
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API. Is the server running?")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def main():
    """Run test queries."""
    
    # Test queries
    queries = [
        "What are cases about division of matrimonial assets?",
        "Find cases involving child custody disputes",
        "What legal principles are mentioned in case 2024_SGHCF_17?",
        "Show me arguments made by appellants in recent cases",
    ]
    
    # If a query is provided as command line argument, use that
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        test_query(query)
    else:
        # Otherwise, run all test queries
        print("\n🚀 Testing Legal RAG API with sample queries...\n")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[Test {i}/{len(queries)}]")
            test_query(query)
            
            if i < len(queries):
                input("\nPress Enter to continue to next query...")
        
        print(f"\n{'='*80}")
        print("✅ All test queries completed!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

