"""
Test script to diagnose embedding generation issues.
"""

import logging
import sys
from src.config import config
from src.tools.embedding_utils import EmbeddingGenerator

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('embedding_test.log')
    ]
)

logger = logging.getLogger(__name__)

def test_embedding_generation():
    """Test embedding generation with detailed diagnostics."""
    
    print("\n" + "="*80)
    print("EMBEDDING GENERATION DIAGNOSTIC TEST")
    print("="*80 + "\n")
    
    # Check configuration
    print("1. Checking Configuration:")
    print(f"   - OPENAI_API_KEY: {'SET' if config.OPENAI_API_KEY else 'NOT SET'}")
    print(f"   - OPENAI_MODEL: {config.OPENAI_MODEL}")
    
    # Try to initialize EmbeddingGenerator
    print("\n2. Initializing EmbeddingGenerator:")
    try:
        embedding_gen = EmbeddingGenerator()
        print("   ✓ EmbeddingGenerator initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize EmbeddingGenerator: {e}")
        return
    
    # Try to generate embedding
    print("\n3. Testing Embedding Generation:")
    test_queries = [
        "How are joint accounts with other parties dealt with?",
        "What is the division of matrimonial assets?",
        "Test query"
    ]
    
    for query in test_queries:
        print(f"\n   Query: {query[:50]}...")
        try:
            embedding = embedding_gen.generate_embedding(query)
            if embedding:
                print(f"   ✓ Generated embedding of dimension {len(embedding)}")
                print(f"   - First 5 values: {embedding[:5]}")
            else:
                print(f"   ✗ Embedding is None")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_embedding_generation()

