"""
Re-ingest the 5 cases that had JSON parsing errors with the improved parser.
"""

import logging
from src.ingestion.graph_ingestion import GraphIngestionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cases with parsing errors that need to be re-ingested
problem_cases = [
    ("case_extract_2025_9", "2025_SGHCF_9"),
    ("case_extract_2025_47", "2025_SGHCF_47"),
    ("case_extract_2025_15", "2025_SGHCF_15"),
    ("case_extract_2024_7", "2024_SGHCF_7"),
]

pipeline = GraphIngestionPipeline()

print("="*80)
print("RE-INGESTING PROBLEM CASES WITH IMPROVED PARSER")
print("="*80)

for case_extract_name, case_number in problem_cases:
    print(f"\n[Re-ingesting] {case_extract_name} → {case_number}")
    print("-"*80)
    
    try:
        pipeline.ingest_case(case_extract_name)
        print(f"✅ Successfully re-ingested {case_extract_name}")
    except Exception as e:
        print(f"❌ Error re-ingesting {case_extract_name}: {e}")
        import traceback
        traceback.print_exc()

pipeline.close()

print("\n" + "="*80)
print("RE-INGESTION COMPLETE")
print("="*80)

