#!/usr/bin/env python3
"""
Test script for the intermediate question generation feature.

Tests the new generate_intermediate_question_node with a complex custody query.
"""

import json
import logging
from src.api.agent import LegalRAGAgent

# Configure logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_intermediate_questions():
    """Test the intermediate question generation with a complex custody query."""
    
    # Initialize the agent
    agent = LegalRAGAgent()
    
    # Test query - complex custody case with overseas elements
    query = "Are there any cases where custody is fought for a child living overseas? Provide some example cases where the father got custody of the child back into Singapore"
    
    logger.info("=" * 100)
    logger.info("TESTING INTERMEDIATE QUESTION GENERATION")
    logger.info("=" * 100)
    logger.info(f"Query: {query}")
    logger.info("=" * 100)
    
    # Run the agent
    result = agent.run(query)
    
    # Extract and display results
    logger.info("\n" + "=" * 100)
    logger.info("FINAL RESULT")
    logger.info("=" * 100)
    
    if isinstance(result, dict):
        # Display answer
        if "answer" in result:
            logger.info(f"\nAnswer:\n{result['answer']}")
        
        # Display workflow nodes
        if "workflow_nodes" in result:
            logger.info(f"\n\nWorkflow Execution ({len(result['workflow_nodes'])} nodes):")
            logger.info("-" * 100)
            
            for node in result["workflow_nodes"]:
                logger.info(f"\nNode: {node.get('node_name', 'unknown')}")
                logger.info(f"  Type: {node.get('node_type', 'unknown')}")
                logger.info(f"  Order: {node.get('execution_order', 'unknown')}")
                logger.info(f"  Results: {node.get('results_count', 0)}")
                logger.info(f"  Summary: {node.get('summary', 'N/A')}")
                
                # Display details for intermediate question node
                if node.get('node_name') == 'generate_intermediate_question_node':
                    details = node.get('details', {})
                    logger.info(f"  Details:")
                    logger.info(f"    - Iteration: {details.get('iteration', 'N/A')}")
                    logger.info(f"    - Questions Generated: {details.get('questions_generated', 0)}")
                    if 'intermediate_questions' in details:
                        logger.info(f"    - Questions:")
                        for idx, q in enumerate(details['intermediate_questions'], 1):
                            logger.info(f"      {idx}. {q}")
        
        # Display retrieved context
        if "retrieved_context" in result:
            logger.info(f"\n\nRetrieved Context ({len(result['retrieved_context'])} sources):")
            logger.info("-" * 100)
            for idx, ctx in enumerate(result["retrieved_context"][:5], 1):
                logger.info(f"\nSource {idx}:")
                logger.info(f"  Type: {ctx.get('type', 'unknown')}")
                logger.info(f"  Chunk ID: {ctx.get('chunk_id', 'unknown')}")
                if 'metadata' in ctx and 'rerank_score' in ctx['metadata']:
                    logger.info(f"  Rerank Score: {ctx['metadata']['rerank_score']:.4f}")
    
    logger.info("\n" + "=" * 100)
    logger.info("TEST COMPLETE")
    logger.info("=" * 100)
    
    return result


if __name__ == "__main__":
    result = test_intermediate_questions()
    
    # Save result to file for inspection
    with open("test_result.json", "w") as f:
        # Convert result to JSON-serializable format
        json_result = {
            "answer": result.get("answer", ""),
            "workflow_nodes": result.get("workflow_nodes", []),
            "retrieved_context_count": len(result.get("retrieved_context", [])),
        }
        json.dump(json_result, f, indent=2)
    
    logger.info("\nResult saved to test_result.json")

