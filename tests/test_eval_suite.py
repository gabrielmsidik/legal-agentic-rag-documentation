"""
Test Evaluation Suite for Agent v3.0

This suite loads QA pairs from qa_for_agent_final.json and evaluates the agent
by sending queries to the API endpoint (localhost:8000/query) and comparing
the agent's answers with expected correct answers.

Usage:
    python -m pytest tests/test_eval_suite.py -v -s

    Or run directly:
    python tests/test_eval_suite.py
"""

import json
import requests
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_QUERY_ENDPOINT = f"{API_BASE_URL}/query"
API_TIMEOUT = 120  # seconds


class EvaluationSuite:
    """Evaluation suite for testing agent against QA dataset."""

    def __init__(self, qa_file_path: str = "tests/qa_for_agent_final.json"):
        """Initialize the evaluation suite."""
        self.qa_file_path = Path(qa_file_path)
        self.qa_data: List[Dict] = []
        self.results: List[Dict] = []
        self.load_qa_data()

    def load_qa_data(self) -> None:
        """Load QA data from JSON file."""
        if not self.qa_file_path.exists():
            raise FileNotFoundError(f"QA file not found: {self.qa_file_path}")

        with open(self.qa_file_path, 'r') as f:
            self.qa_data = json.load(f)

        logger.info(f"Loaded {len(self.qa_data)} QA pairs from {self.qa_file_path}")

    def extract_answer_from_response(self, response_text: str) -> str:
        """
        Extract the answer letter (A, B, C, D, E) from the agent's response.

        The agent should respond with just the letter, but we'll be lenient
        and extract it if it's embedded in the response.
        """
        # Look for single letter answer (A, B, C, D, E)
        match = re.search(r'\b([A-E])\b', response_text.strip())
        if match:
            return match.group(1)

        # If no match, return the first character if it's a valid answer
        first_char = response_text.strip()[0].upper() if response_text.strip() else ""
        if first_char in ['A', 'B', 'C', 'D', 'E']:
            return first_char

        return f"INVALID - {response_text}"

    def query_agent(self, question: str) -> Tuple[str, Dict]:
        """
        Send a query to the agent API and get the response.

        Returns:
            Tuple of (answer_text, full_response_dict)
        """
        try:
            payload = {"query": question}
            response = requests.post(
                API_QUERY_ENDPOINT,
                json=payload,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()
            answer_text = data.get("answer", "")

            logger.info(f"Agent response: {answer_text[:100]}...")

            return answer_text, data

        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error: Could not connect to {API_QUERY_ENDPOINT}")
            return "CONNECTION_ERROR", {}
        except requests.exceptions.Timeout:
            logger.error(f"Timeout: Request to {API_QUERY_ENDPOINT} timed out")
            return "TIMEOUT", {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return "REQUEST_ERROR", {}
        except json.JSONDecodeError:
            logger.error("Response is not valid JSON")
            return "JSON_ERROR", {}

    def evaluate_single_qa(self, qa_item: Dict, index: int) -> Dict:
        """Evaluate a single QA pair."""
        qa_id = qa_item.get("id", index)
        question = qa_item.get("question", "")
        correct_answer = qa_item.get("correct_answer", "")

        logger.info(f"\n[{index + 1}/{len(self.qa_data)}] Evaluating QA ID {qa_id}")
        logger.info(f"Question: {question[:80]}...")

        # Query the agent
        agent_response, full_response = self.query_agent(question)

        # Extract answer from response
        extracted_answer = self.extract_answer_from_response(agent_response)

        # Determine if correct
        is_correct = extracted_answer == correct_answer

        # Build result
        result = {
            "qa_id": qa_id,
            "question": question,
            "correct_answer": correct_answer,
            "agent_response": agent_response[:200],  # Truncate for storage
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "full_response": full_response,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Expected: {correct_answer}, Got: {extracted_answer}, "
                   f"Correct: {is_correct}")

        return result

    def run_evaluation(self, limit: int = None) -> None:
        """
        Run evaluation on all QA pairs.

        Args:
            limit: Maximum number of QA pairs to evaluate (None = all)
        """
        qa_items = self.qa_data[:limit] if limit else self.qa_data

        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Evaluation Suite - {len(qa_items)} questions")
        logger.info(f"API Endpoint: {API_QUERY_ENDPOINT}")
        logger.info(f"{'='*80}\n")

        for index, qa_item in enumerate(qa_items):
            try:
                result = self.evaluate_single_qa(qa_item, index)
                self.results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating QA {qa_item.get('id', index)}: {e}")
                self.results.append({
                    "qa_id": qa_item.get("id", index),
                    "question": qa_item.get("question", ""),
                    "correct_answer": qa_item.get("correct_answer", ""),
                    "error": str(e),
                    "is_correct": False
                })

        self.print_summary()

    def print_summary(self) -> None:
        """Print evaluation summary."""
        if not self.results:
            logger.warning("No results to summarize")
            return

        total = len(self.results)
        correct = sum(1 for r in self.results if r.get("is_correct", False))
        accuracy = (correct / total * 100) if total > 0 else 0

        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total Questions: {total}")
        logger.info(f"Correct Answers: {correct}")
        logger.info(f"Incorrect Answers: {total - correct}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"{'='*80}\n")

        # Print incorrect answers
        incorrect = [r for r in self.results if not r.get("is_correct", False)]
        if incorrect:
            logger.info(f"Incorrect Answers ({len(incorrect)}):")
            for r in incorrect:
                logger.info(f"  QA {r['qa_id']}: Expected {r['correct_answer']}, "
                           f"Got {r.get('extracted_answer', 'N/A')}")

    def save_results(self, output_file: str = "eval_results.json") -> None:
        """Save evaluation results to JSON file."""
        output_path = Path(output_file)

        # Prepare results for JSON serialization
        serializable_results = []
        for r in self.results:
            result_copy = r.copy()
            # Remove full_response as it may contain non-serializable objects
            result_copy.pop("full_response", None)
            serializable_results.append(result_copy)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def test_agent_evaluation_suite():
    """Pytest test function for the evaluation suite."""
    suite = EvaluationSuite()
    suite.run_evaluation(limit=5)  # Run on first 5 questions for quick test

    # Assert that we got results
    assert len(suite.results) > 0, "No results generated"

    # Assert that at least some questions were answered
    answered = sum(1 for r in suite.results
                  if r.get("extracted_answer") not in ["INVALID", "CONNECTION_ERROR"])
    assert answered > 0, "No questions were answered by the agent"


if __name__ == "__main__":
    # Run the full evaluation suite
    suite = EvaluationSuite()
    suite.run_evaluation()  # Run all questions
    suite.save_results("eval_results.json")
