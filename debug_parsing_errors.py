"""
Debug script to investigate JSON parsing errors for specific cases.
"""

import logging
from src.ingestion.graph_ingestion import GraphIngestionPipeline

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Cases with parsing errors
problem_cases = [
    "case_extract_2025_9",
    "case_extract_2025_47",
    "case_extract_2025_15",
    "case_extract_2024_7",
]

pipeline = GraphIngestionPipeline()

for case_extract_name in problem_cases:
    print(f"\n{'='*80}")
    print(f"Testing: {case_extract_name}")
    print(f"{'='*80}")
    
    try:
        # Fetch case text
        case_text, chunks = pipeline.fetch_case_extract_from_pinecone(case_extract_name)
        print(f"✓ Fetched {len(case_text)} characters")
        
        # Extract case number
        case_number = case_extract_name.replace("case_extract_", "").replace("_", "_SGHCF_")
        
        # Try second pass extraction (where errors occur)
        print(f"\nAttempting second pass extraction...")
        result = pipeline.extract_second_pass(case_text, case_number)
        
        if result and result.get("arguments"):
            print(f"✓ Successfully extracted {len(result['arguments'])} arguments")
        else:
            print(f"✗ Failed to extract arguments (result: {result})")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

pipeline.close()
print("\n" + "="*80)
print("Debug complete")

