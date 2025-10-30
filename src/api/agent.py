"""
LangGraph agent implementation for the Legal Agentic RAG system.
Implements a state machine with planning, tool execution, and synthesis nodes.
"""

import logging
from typing import List, Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from src.config import config
from src.models.schemas import GraphState, Source
from src.tools import get_vector_search_tool, get_graph_search_tool, get_reranker

logger = logging.getLogger(__name__)


class LegalRAGAgent:
    """
    LangGraph-based agent for answering legal queries using hybrid vector + graph search.
    """
    
    def __init__(self):
        """Initialize the agent with LLM and tools."""
        self.llm = ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY
        )

        self.vector_tool = get_vector_search_tool()
        self.graph_tool = get_graph_search_tool()
        self.reranker_tool = get_reranker()

        # Build the state graph
        self.graph = self._build_graph()
        logger.info("Initialized LegalRAGAgent")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        # Create the state graph
        workflow = StateGraph(GraphState)

        workflow.add_node("plan_step", self.plan_step)
        workflow.add_node("vector_search_node", self.vector_search_node)
        workflow.add_node("graph_search_node", self.graph_search_node)
        workflow.add_node("rerank_results_node", self.rerank_results_node)
        workflow.add_node("evaluate_results_node", self.evaluate_results_node)
        workflow.add_node("synthesize_answer_node", self.synthesize_answer_node)

        # Set entry point
        workflow.set_entry_point("plan_step")

        # Add conditional edges from plan_step
        workflow.add_conditional_edges(
            "plan_step",
            self.route_action,
            {
                "vector": "vector_search_node",
                "graph": "graph_search_node",
            }
        )

        # Add edges from tool nodes to reranking
        workflow.add_edge("vector_search_node", "rerank_results_node")
        workflow.add_edge("graph_search_node", "rerank_results_node")

        # Add edge from reranking to evaluation
        workflow.add_edge("rerank_results_node", "evaluate_results_node")

        # Add conditional edges from evaluation (v3.0: direct to plan_step or synthesize)
        workflow.add_conditional_edges(
            "evaluate_results_node",
            self.should_continue,
            {
                "continue": "plan_step",  # v3.0: go back to planning instead of intermediate questions
                "synthesize": "synthesize_answer_node",
            }
        )

        # Add edge from synthesis to end
        workflow.add_edge("synthesize_answer_node", END)

        # Compile the graph
        return workflow.compile()
    
    def plan_step(self, state: GraphState) -> GraphState:
        """
        LLM node: Analyze the query and decide the next action.
        """
        logger.info("=" * 80)
        logger.info("EXECUTING PLAN_STEP")
        logger.info("=" * 80)

        original_query = state["original_query"]
        retrieved_context = state.get("retrieved_context", [])
        iteration_count = state.get("iteration_count", 0)
        chain_of_thought = state.get("chain_of_thought", [])
        workflow_nodes = state.get("workflow_nodes", [])

        current_query = original_query

        # Build detailed context summary
        context_summary = self._build_context_summary(retrieved_context, [])

        # Create planning prompt with graph schema and context
        system_prompt = """You are a strategic planning agent for a legal research system
Your job is to analyze the current situation and generate a strategy for the next search action.

KNOWLEDGE GRAPH SCHEMA:
Node Types: Case, Person, Statute, LegalPrinciple, Argument
Key Relationships: PRESIDED_OVER, IS_PARTY_IN, REPRESENTED, CITES, CONTAINS, IS_RELATED_TO, MADE, IS_ABOUT, REFERENCES

AVAILABLE SEARCH TYPES:
1. "vector" - Semantic search on Pinecone (best for: nuanced queries, semantic understanding)
2. "graph" - Structured search on Neo4j (best for: entity relationships, precedents, specific details)

SEARCH APPROACHES:
- "semantic": Use semantic similarity for finding related concepts
- "keyword": Use keyword matching for specific terms
- "principles": Search for legal principles and concepts
- "combined": Mix multiple approaches

YOUR TASK:
1. Analyze the current query and retrieved context
2. Identify what information is still needed
3. Generate a strategy with:
   - search_type: "vector" or "graph"
   - approach: "semantic", "keyword", "principles", or "combined"
   - query: Refined query string for the search
   - focus_areas: List of key areas to focus on
   - filters: List of queries/cases to avoid (dead-ends)
   - reasoning: Brief explanation of why this strategy

Respond with ONLY a JSON object (no markdown, no extra text).
"""

        user_prompt = f"""Current Query: {current_query}

{context_summary}

Chain of Thought History:
{self._format_chain_of_thought(chain_of_thought)}

Generate a strategy for the next search. Respond with ONLY a JSON object."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            strategy_text = response.content.strip()

            # Parse strategy JSON
            import json
            strategy = json.loads(strategy_text)
            plan = strategy.get("search_type", "vector")

            logger.info(f"Generated strategy: {plan}")
            logger.info(f"Strategy details: {strategy}")

        except Exception as e:
            logger.warning(f"Failed to parse strategy JSON: {e}. Using default strategy.")
            # Default strategy
            if len(retrieved_context) == 0:
                plan = "vector"
                strategy = {
                    "search_type": "vector",
                    "approach": "semantic",
                    "query": current_query,
                    "focus_areas": ["main query"],
                    "filters": [],
                    "reasoning": "Initial search - start with vector search for semantic understanding"
                }
            else:
                plan = "graph"
                strategy = {
                    "search_type": "graph",
                    "approach": "semantic",
                    "query": current_query,
                    "focus_areas": ["relationships", "precedents"],
                    "filters": [],
                    "reasoning": "Follow-up search - use graph search for structured relationships"
                }

        # Update chain of thought with planning analysis
        cot_entry = f"Iteration {iteration_count + 1} - Planning:\n"
        cot_entry += f"  Query: {current_query[:80]}\n"
        cot_entry += f"  Strategy: {plan} search using {strategy.get('approach', 'unknown')} approach\n"
        cot_entry += f"  Focus Areas: {', '.join(strategy.get('focus_areas', []))}\n"
        cot_entry += f"  Filters: {', '.join(strategy.get('filters', [])) if strategy.get('filters') else 'None'}"
        chain_of_thought.append(cot_entry)

        # Create workflow node entry for plan_step
        execution_order = len(workflow_nodes) + 1
        workflow_node = {
            "node_name": "plan_step",
            "node_type": "planning",
            "results_count": 0,
            "summary": f"Generated strategy for {plan} search. Focus: {', '.join(strategy.get('focus_areas', [])[:2])}",
            "execution_order": execution_order,
            "details": {
                "plan": plan,
                "strategy": strategy,
                "current_query": current_query[:100] + "..." if len(current_query) > 100 else current_query,
                "iteration": iteration_count + 1,
                "context_sources": len(retrieved_context),
                "context_types": self._get_context_types(retrieved_context)
            }
        }
        workflow_nodes.append(workflow_node)

        state["plan"] = plan
        state["strategy"] = strategy
        state["current_query"] = current_query
        state["chain_of_thought"] = chain_of_thought
        state["iteration_count"] = iteration_count + 1
        state["workflow_nodes"] = workflow_nodes

        logger.info("=" * 80)
        logger.info("PLAN_STEP COMPLETE")
        logger.info("=" * 80)

        return state

    def _format_chain_of_thought(self, chain_of_thought: List[str]) -> str:
        """Format chain of thought for display in prompts."""
        if not chain_of_thought:
            return "No previous iterations yet."
        return "\n".join(chain_of_thought[-3:])  # Show last 3 entries

    def _build_context_summary(self, retrieved_context: List[Dict], intermediate_steps: List[tuple]) -> str:
        """Build a detailed summary of current context for the planning prompt."""
        summary_parts = []

        # Overall context count
        summary_parts.append(f"Retrieved {len(retrieved_context)} sources so far.")

        # Break down by source type
        if retrieved_context:
            vector_count = sum(1 for ctx in retrieved_context if ctx.get("type") == "vector")
            graph_count = sum(1 for ctx in retrieved_context if ctx.get("type") == "graph")

            if vector_count > 0:
                summary_parts.append(f"  - Vector search results: {vector_count}")
            if graph_count > 0:
                summary_parts.append(f"  - Graph search results: {graph_count}")

            # Show sample of retrieved information
            if len(retrieved_context) > 0:
                summary_parts.append("\nCurrent context includes:")
                for ctx in retrieved_context[:3]:
                    if ctx.get("type") == "vector":
                        logger.info(f"Vector context: {ctx.get('data', '')}...")
                        summary_parts.append(f"  - Vector: {ctx.get('data', '')}...")
                    else:
                        logger.info(f"Context Type: {ctx.get('type')}")
                        logger.info(f"Context (Above should be graph): {ctx.get('data', '')}")
                        summary_parts.append(f"  - Graph: {ctx.get('data', '')}...")
                if len(retrieved_context) > 3:
                    summary_parts.append(f"  ... and {len(retrieved_context) - 3} more sources")

        # Previous actions
        if intermediate_steps:
            recent_actions = [step[0] for step in intermediate_steps[-3:]]
            summary_parts.append(f"\nPrevious actions: {recent_actions}")

        return "\n".join(summary_parts)

    def _get_context_types(self, retrieved_context: List[Dict]) -> Dict[str, int]:
        """Get breakdown of context types."""
        types = {"vector": 0, "graph": 0}
        for ctx in retrieved_context:
            ctx_type = ctx.get("type", "unknown")
            if ctx_type in types:
                types[ctx_type] += 1
        return types
    
    def route_action(self, state: GraphState) -> Literal["vector", "graph", "synthesize"]:
        """
        Conditional logic node: Route based on the plan.
        """
        plan = state.get("plan", "synthesize")
        
        if "vector" in plan:
            return "vector"
        elif "graph" in plan:
            return "graph"
        else:
            return "synthesize"
    
    def _clean_context_for_storage(self, ctx_dict: dict) -> dict:
        """
        Clean context dictionary before storing in retrieved_context.
        Removes embedding vectors and other unnecessary fields to minimize token usage.

        Args:
            ctx_dict: Context dictionary from model_dump()

        Returns:
            Cleaned context dictionary
        """
        cleaned = ctx_dict.copy()

        # Remove embedding vectors from data if it's a dict
        if isinstance(cleaned.get("data"), dict):
            data = cleaned["data"].copy()
            # Remove all embedding fields
            fields_to_remove = [
                "summary_embedding", "outcome_embedding", "judge_reasoning_embedding",
                "embedding", "embeddings", "vector", "vectors"
            ]
            for field in fields_to_remove:
                data.pop(field, None)
            cleaned["data"] = data

        # Remove unnecessary metadata fields
        if "metadata" in cleaned:
            metadata = cleaned["metadata"].copy()
            # Keep only essential metadata
            essential_keys = {"case_extract_name", "chunk_id", "score", "relevance_score", "vector_chunk_ids"}
            cleaned["metadata"] = {k: v for k, v in metadata.items() if k in essential_keys or k.startswith("case_")}

        return cleaned

    def vector_search_node(self, state: GraphState) -> GraphState:
        """
        Tool node: Execute vector search
        """
        logger.info("Executing vector_search_node")

        original_query = state["original_query"]
        current_query = state.get("current_query", original_query)
        retrieved_context = state.get("retrieved_context", [])
        chain_of_thought = state.get("chain_of_thought", [])
        iteration_count = state.get("iteration_count", 0)
        workflow_nodes = state.get("workflow_nodes", [])

        logger.info(f"Search query: {current_query[:80]}...")
        logger.info(f"Current context size before vector search: {len(retrieved_context)} sources")

        # Perform vector search
        results = self.vector_tool.search(
            query=current_query,
            top_k=config.VECTOR_SEARCH_TOP_K
        )

        # Add cleaned results to context with deduplication
        existing_chunk_ids = set()
        for ctx in retrieved_context:
            if ctx.get("chunk_id"):
                existing_chunk_ids.add(ctx["chunk_id"])

        logger.info(f"Existing chunk IDs in context: {len(existing_chunk_ids)}")

        added_count = 0
        skipped_count = 0
        for result in results:
            chunk_id = result.chunk_id
            if chunk_id and chunk_id in existing_chunk_ids:
                logger.debug(f"Skipping duplicate vector source: chunk_id={chunk_id}")
                skipped_count += 1
            else:
                cleaned_ctx = self._clean_context_for_storage(result.model_dump())
                retrieved_context.append(cleaned_ctx)
                if chunk_id:
                    existing_chunk_ids.add(chunk_id)
                added_count += 1

        logger.info(f"Vector search: added {added_count} new sources, skipped {skipped_count} duplicates")
        logger.info(f"Context size after vector search: {len(retrieved_context)} sources")

        # Update chain of thought with search results
        cot_entry = f"Iteration {iteration_count} - Vector Search:\n"
        cot_entry += f"  Query: {current_query[:80]}\n"
        cot_entry += f"  Results: {len(results)} found, {added_count} added, {skipped_count} duplicates\n"
        if results:
            top_cases = set()
            for r in results[:3]:
                case_name = r.metadata.get("case_extract_name", "unknown")
                if case_name:
                    top_cases.add(case_name)
            cot_entry += f"  Key Cases: {', '.join(list(top_cases)[:2])}\n"
            cot_entry += f"  Top Score: {results[0].metadata.get('score', 0):.4f}"
        else:
            cot_entry += f"  Key Cases: None\n"
            cot_entry += f"  Status: No results found"
        chain_of_thought.append(cot_entry)

        # Create workflow node entry
        execution_order = len(workflow_nodes) + 1
        workflow_node = {
            "node_name": "vector_search_node",
            "node_type": "tool",
            "results_count": len(results),
            "summary": f"Performed vector search on Pinecone. Found {len(results)} relevant case chunks using semantic similarity. Top result similarity score: {results[0].metadata.get('score', 0):.4f}" if results else "Vector search returned no results",
            "execution_order": execution_order,
            "details": {
                "search_type": "semantic",
                "database": "Pinecone",
                "top_k": config.VECTOR_SEARCH_TOP_K,
                "results": [
                    {
                        "case_extract": r.metadata.get("case_extract_name", "unknown"),
                        "chunk_id": r.chunk_id,
                        "similarity_score": r.metadata.get("score", 0)
                    }
                    for r in results
                ]
            }
        }
        workflow_nodes.append(workflow_node)

        state["retrieved_context"] = retrieved_context
        state["chain_of_thought"] = chain_of_thought
        state["workflow_nodes"] = workflow_nodes

        logger.info(f"Vector search returned {len(results)} results")

        return state
    
    def graph_search_node(self, state: GraphState) -> GraphState:
        """
        Tool node: Execute hybrid graph search (semantic + structured)

        Performs both semantic search (using embeddings) and structured search
        (using relationships and keywords) to provide comprehensive results.

        - Uses strategy from plan_step to guide search approach
        - Updates chain_of_thought with search results and findings
        """
        logger.info("Executing graph_search_node with hybrid search")

        original_query = state["original_query"]
        current_query = state.get("current_query", original_query)
        retrieved_context = state.get("retrieved_context", [])
        chain_of_thought = state.get("chain_of_thought", [])
        iteration_count = state.get("iteration_count", 0)
        workflow_nodes = state.get("workflow_nodes", [])

        logger.info(f"Search query: {current_query[:80]}...")
        logger.info(f"Current context size before graph search: {len(retrieved_context)} sources")

        # Perform semantic search first (using embeddings)
        logger.info("Performing semantic search on graph embeddings...")
        semantic_results = self.graph_tool.semantic_graph_search(query=current_query, top_k=5)
        logger.info(f"Semantic search returned {len(semantic_results)} results")

        # Perform structured search (using relationships and keywords)
        logger.info("Performing structured search on graph relationships...")
        structured_results = self.graph_tool.natural_language_search(query=current_query)
        logger.info(f"Structured search returned {len(structured_results)} results")

        # Deduplicate within semantic results
        dedup_semantic = self._deduplicate_graph_results(semantic_results)
        logger.info(f"After deduplication within semantic results: {len(dedup_semantic)} unique results")

        # Deduplicate within structured results
        dedup_structured = self._deduplicate_graph_results(structured_results)
        logger.info(f"After deduplication within structured results: {len(dedup_structured)} unique results")

        # Combine deduplicated results
        combined_results = dedup_semantic + dedup_structured

        # Deduplicate combined results (remove duplicates between semantic and structured)
        deduplicated_results = self._deduplicate_graph_results(combined_results)
        logger.info(f"After deduplication between semantic and structured: {len(deduplicated_results)} unique results")

        # Build set of existing graph node identifiers to avoid duplicates
        existing_graph_ids = set()
        for ctx in retrieved_context:
            if ctx.get("type") == "graph" and isinstance(ctx.get("data"), dict):
                node_data = ctx["data"]
                # Extract unique identifier for graph nodes
                case_num = node_data.get('case_number') or node_data.get('c.case_number')
                if case_num:
                    existing_graph_ids.add(('Case', case_num))
                else:
                    statute_name = node_data.get('name') or node_data.get('s.name')
                    statute_section = node_data.get('section') or node_data.get('s.section')
                    if statute_name and statute_section:
                        existing_graph_ids.add(('Statute', statute_name, statute_section))

        logger.info(f"Existing graph node IDs in context: {len(existing_graph_ids)}")

        # Add cleaned results to context with deduplication
        chunk_ids_to_fetch = []
        graph_added = 0
        graph_skipped = 0
        for result in deduplicated_results:
            node_data = result.data if isinstance(result.data, dict) else {}

            # Check if this graph node already exists
            case_num = node_data.get('case_number') or node_data.get('c.case_number')
            if case_num:
                node_id = ('Case', case_num)
            else:
                statute_name = node_data.get('name') or node_data.get('s.name')
                statute_section = node_data.get('section') or node_data.get('s.section')
                if statute_name and statute_section:
                    node_id = ('Statute', statute_name, statute_section)
                else:
                    node_id = None

            if node_id and node_id in existing_graph_ids:
                logger.debug(f"Skipping duplicate graph node: {node_id}")
                graph_skipped += 1
            else:
                cleaned_ctx = self._clean_context_for_storage(result.model_dump())
                retrieved_context.append(cleaned_ctx)
                if node_id:
                    existing_graph_ids.add(node_id)
                graph_added += 1
                # Check if result contains vector_chunk_ids
                if result.metadata.get("vector_chunk_ids"):
                    chunk_ids_to_fetch.extend(result.metadata["vector_chunk_ids"])

        logger.info(f"Graph search: added {graph_added} new sources, skipped {graph_skipped} duplicates")

        # If we found chunk IDs, fetch them
        vector_chunks_fetched = 0
        if chunk_ids_to_fetch:
            logger.info(f"Fetching {len(chunk_ids_to_fetch)} vector chunks from graph results")
            vector_results = self.vector_tool.fetch_by_ids(chunk_ids_to_fetch)
            vector_chunks_fetched = len(vector_results)

            # Deduplicate vector chunks against existing chunk IDs
            existing_chunk_ids = set()
            for ctx in retrieved_context:
                if ctx.get("chunk_id"):
                    existing_chunk_ids.add(ctx["chunk_id"])

            vector_added = 0
            vector_skipped = 0
            for result in vector_results:
                if result.chunk_id and result.chunk_id in existing_chunk_ids:
                    logger.debug(f"Skipping duplicate vector chunk: {result.chunk_id}")
                    vector_skipped += 1
                else:
                    cleaned_ctx = self._clean_context_for_storage(result.model_dump())
                    retrieved_context.append(cleaned_ctx)
                    if result.chunk_id:
                        existing_chunk_ids.add(result.chunk_id)
                    vector_added += 1

            logger.info(f"Vector chunks: added {vector_added} new sources, skipped {vector_skipped} duplicates")
            vector_chunks_fetched = vector_added

        logger.info(f"Context size after graph search: {len(retrieved_context)} sources")

        # Update chain of thought with search results
        cot_entry = f"Iteration {iteration_count} - Graph Search:\n"
        cot_entry += f"  Query: {current_query[:80]}\n"
        cot_entry += f"  Results: Semantic {len(semantic_results)} → {len(dedup_semantic)} dedup, Structured {len(structured_results)} → {len(dedup_structured)} dedup\n"
        cot_entry += f"  Added: {graph_added} graph nodes, {vector_chunks_fetched} vector chunks\n"
        if deduplicated_results:
            top_nodes = []
            for r in deduplicated_results[:2]:
                if isinstance(r.data, dict):
                    case_num = r.data.get('case_number') or r.data.get('c.case_number')
                    if case_num:
                        top_nodes.append(case_num)
            cot_entry += f"  Key Nodes: {', '.join(top_nodes) if top_nodes else 'Various'}"
        else:
            cot_entry += f"  Key Nodes: None"
        chain_of_thought.append(cot_entry)

        # Create workflow node entry with detailed deduplication info
        execution_order = len(workflow_nodes) + 1
        workflow_node = {
            "node_name": "graph_search_node",
            "node_type": "tool",
            "results_count": len(deduplicated_results),
            "summary": f"Performed hybrid graph search on Neo4j. Semantic search: {len(semantic_results)} results → {len(dedup_semantic)} after dedup. Structured search: {len(structured_results)} results → {len(dedup_structured)} after dedup. Combined: {len(deduplicated_results)} unique results. Fetched {vector_chunks_fetched} additional vector chunks.",
            "execution_order": execution_order,
            "details": {
                "search_type": "hybrid",
                "database": "Neo4j",
                "semantic_results_raw": len(semantic_results),
                "semantic_results_dedup": len(dedup_semantic),
                "structured_results_raw": len(structured_results),
                "structured_results_dedup": len(dedup_structured),
                "final_results": len(deduplicated_results),
                "vector_chunks_fetched": vector_chunks_fetched
            }
        }
        workflow_nodes.append(workflow_node)

        state["retrieved_context"] = retrieved_context
        state["chain_of_thought"] = chain_of_thought
        state["workflow_nodes"] = workflow_nodes

        logger.info(f"Hybrid graph search returned {len(deduplicated_results)} results")

        return state

    def evaluate_results_node(self, state: GraphState) -> GraphState:
        """
        LLM node: Assess if current context is sufficient to answer the query.

        Improved evaluation logic:
        1. Requires both vector and graph search before synthesizing (for diversity)
        2. Checks relevance of results to the query (on already-reranked top-5 sources)
        3. Only synthesizes when we have sufficient RELEVANT results

        NOTE: This node now receives already-reranked context (top 5 sources from rerank_results_node)
        """
        logger.info("=" * 80)
        logger.info("EXECUTING EVALUATE_RESULTS_NODE")
        logger.info("=" * 80)

        original_query = state["original_query"]
        retrieved_context = state.get("retrieved_context", [])
        iteration_count = state.get("iteration_count", 0) + 1
        intermediate_steps = state.get("intermediate_steps", [])
        workflow_nodes = state.get("workflow_nodes", [])

        # Track evaluation decision
        evaluation_reason = ""
        decision_made = ""

        # Check iteration limit
        if iteration_count >= config.MAX_ITERATIONS:
            logger.info(f"Current iterations {iteration_count}/{config.MAX_ITERATIONS}")
            logger.warning(f"Reached max iterations ({config.MAX_ITERATIONS}), forcing synthesis")
            state["needs_more_info"] = False
            evaluation_reason = f"Reached max iterations ({config.MAX_ITERATIONS}), forcing synthesis"
            decision_made = "synthesize"

            # Add workflow node for evaluation
            execution_order = len(workflow_nodes) + 1
            workflow_node = {
                "node_name": "evaluate_results_node",
                "node_type": "decision",
                "results_count": len(retrieved_context),
                "summary": f"Evaluated context sufficiency. {evaluation_reason}",
                "execution_order": execution_order,
                "details": {
                    "decision": decision_made,
                    "reason": evaluation_reason,
                    "context_count": len(retrieved_context),
                    "iteration_count": iteration_count
                }
            }
            workflow_nodes.append(workflow_node)
            state["workflow_nodes"] = workflow_nodes
            return state

        # If no context, need more info
        if not retrieved_context:
            state["needs_more_info"] = True
            evaluation_reason = "No context retrieved yet"
            decision_made = "continue"
            logger.info("No context retrieved, need more info")

            # Add workflow node for evaluation
            execution_order = len(workflow_nodes) + 1
            workflow_node = {
                "node_name": "evaluate_results_node",
                "node_type": "decision",
                "results_count": 0,
                "summary": f"Evaluated context sufficiency. {evaluation_reason}",
                "execution_order": execution_order,
                "details": {
                    "decision": decision_made,
                    "reason": evaluation_reason,
                    "context_count": 0,
                    "iteration_count": iteration_count
                }
            }
            workflow_nodes.append(workflow_node)
            state["workflow_nodes"] = workflow_nodes
            return state

        # Get search types that have been tried
        search_types = [step[0] for step in intermediate_steps]
        logger.info(f"Search types tried: {search_types}")
        has_vector_search = "vector_search" in search_types
        has_graph_search = "graph_search_hybrid" in search_types

        # Also check what types of sources were actually retrieved
        source_types = set()
        for ctx in retrieved_context:
            source_type = ctx.get("type", "unknown")
            source_types.add(source_type)

        logger.info(f"source_types: {source_types}")

        has_vector_sources = "vector" in source_types
        has_graph_sources = "graph" in source_types

        logger.info(f"Search types tried: vector={has_vector_search}, graph={has_graph_search}")
        logger.info(f"Source types retrieved: vector={has_vector_sources}, graph={has_graph_sources}")
        logger.info(f"Retrieved {len(retrieved_context)} sources total")

        # IMPORTANT: Require both search types for diversity
        # Check if we have both search types ATTEMPTED (not just retrieved)
        # This ensures we try both approaches even if one returns no results
        if not (has_vector_search and has_graph_search):
            state["needs_more_info"] = True
            evaluation_reason = f"Need both search types for diversity. Tried: vector={has_vector_search}, graph={has_graph_search}"
            decision_made = "continue"
            logger.info("Need to try both search types for diverse results")

            # Add workflow node for evaluation
            execution_order = len(workflow_nodes) + 1
            workflow_node = {
                "node_name": "evaluate_results_node",
                "node_type": "decision",
                "results_count": len(retrieved_context),
                "summary": f"Evaluated context sufficiency. {evaluation_reason}",
                "execution_order": execution_order,
                "details": {
                    "decision": decision_made,
                    "reason": evaluation_reason,
                    "context_count": len(retrieved_context),
                    "search_types_tried": {"vector": has_vector_search, "graph": has_graph_search},
                    "source_types_retrieved": {"vector": has_vector_sources, "graph": has_graph_sources},
                    "iteration_count": iteration_count
                }
            }
            workflow_nodes.append(workflow_node)
            state["workflow_nodes"] = workflow_nodes
            return state

        # Both search types data retrieved, evaluate relevance
        # Build cleaned context summary (minimal tokens)
        context_summary = f"We have {len(retrieved_context)} sources from both vector and graph search:\n\n"
        for idx, ctx in enumerate(retrieved_context):
            cleaned = self._extract_key_context(ctx)
            if cleaned:
                context_summary += f"- Source {idx+1}: {cleaned}\n"

        # Ask LLM to evaluate relevance
        system_prompt = """You are evaluating whether we have sufficient RELEVANT information to answer a user's query.

            Review the sources and assess:
            1. Do the sources directly address the query?
            2. Are there enough specific examples or case law?
            3. Is the information comprehensive enough?

            Be strict - only say "sufficient" if the sources clearly answer the query with specific details.
            Respond with ONLY "sufficient" or "insufficient"."""

        user_prompt = f"""Query: {original_query}

            Sources Retrieved:
            {context_summary}

            Do we have sufficient RELEVANT information to answer this query?"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        decision = response.content.strip().lower()

        needs_more = "insufficient" in decision
        state["needs_more_info"] = needs_more

        evaluation_reason = f"LLM relevance evaluation: {decision}"
        decision_made = "continue" if needs_more else "synthesize"

        logger.info(f"Relevance evaluation: {'needs more info' if needs_more else 'sufficient'}")

        # Add workflow node for evaluation
        execution_order = len(workflow_nodes) + 1
        workflow_node = {
            "node_name": "evaluate_results_node",
            "node_type": "decision",
            "results_count": len(retrieved_context),
            "summary": f"Evaluated context relevance. {evaluation_reason}. Decision: {decision_made}",
            "execution_order": execution_order,
            "details": {
                "decision": decision_made,
                "reason": evaluation_reason,
                "context_count": len(retrieved_context),
                "search_types_tried": {"vector": has_vector_search, "graph": has_graph_search},
                "source_types_retrieved": {"vector": has_vector_sources, "graph": has_graph_sources},
                "iteration_count": iteration_count,
                "llm_decision": decision
            }
        }
        workflow_nodes.append(workflow_node)
        state["workflow_nodes"] = workflow_nodes

        logger.info("=" * 80)
        logger.info(f"EVALUATE_RESULTS_NODE COMPLETE - Decision: {decision_made}")
        logger.info("=" * 80)

        return state

    def should_continue(self, state: GraphState) -> Literal["continue", "synthesize"]:
        """
        Conditional logic node: Decide whether to continue searching or synthesize.
        """
        needs_more = state.get("needs_more_info", False)
        return "continue" if needs_more else "synthesize"

    def rerank_results_node(self, state: GraphState) -> GraphState:
        """
        Tool node: Rerank vector sources only using Pinecone's reranking API
        - Only reranks vector sources (graph sources are kept as-is)
        - Separates vector and graph sources
        - Keeps top 5 vector sources + all graph sources
        """
        logger.info("=" * 80)
        logger.info("EXECUTING RERANK_RESULTS_NODE")
        logger.info("=" * 80)

        original_query = state["original_query"]
        retrieved_context = state.get("retrieved_context", [])
        workflow_nodes = state.get("workflow_nodes", [])

        # Convert context dicts back to Source objects
        sources = [Source(**ctx) for ctx in retrieved_context]

        # Separate vector and graph sources
        vector_sources = [s for s in sources if s.type == "vector"]
        graph_sources = [s for s in sources if s.type == "graph"]

        logger.info(f"BEFORE RERANKING:")
        logger.info(f"  Total sources: {len(sources)}")
        logger.info(f"  - Vector sources: {len(vector_sources)}")
        logger.info(f"  - Graph sources: {len(graph_sources)}")
        logger.info(f"Query: {original_query[:100]}...")

        # Rerank only vector sources and keep top 5
        reranked_vector = []
        if vector_sources:
            reranked_vector = self.reranker_tool.rerank_sources(
                query=original_query,
                sources=vector_sources,
                top_n=5
            )

        # Combine reranked vector sources with all graph sources
        reranked_sources = reranked_vector + graph_sources

        logger.info(f"AFTER RERANKING (Vector Only):")
        logger.info(f"  Total sources: {len(reranked_sources)}")
        logger.info(f"  - Vector sources (reranked): {len(reranked_vector)}")
        logger.info(f"  - Graph sources (kept as-is): {len(graph_sources)}")

        # Calculate impact metrics for vector sources only
        vector_removed = len(vector_sources) - len(reranked_vector)
        vector_removal_rate = (vector_removed / len(vector_sources) * 100) if vector_sources else 0

        logger.info(f"RERANKING IMPACT (Vector Only):")
        logger.info(f"  Vector sources removed: {vector_removed} ({vector_removal_rate:.1f}%)")
        logger.info(f"  Vector context reduction: {len(vector_sources)} → {len(reranked_vector)} sources")

        # Log rerank scores for vector sources
        if reranked_vector:
            scores = [s.metadata.get("rerank_score", 0) for s in reranked_vector]
            avg_score = sum(scores) / len(scores)
            logger.info(f"RERANK SCORES (Vector):")
            logger.info(f"  Average score: {avg_score:.4f}")
            logger.info(f"  Min score: {min(scores):.4f}")
            logger.info(f"  Max score: {max(scores):.4f}")

            # Log individual scores
            logger.info(f"RANKED VECTOR SOURCES (top 5):")
            for rank, source in enumerate(reranked_vector, 1):
                score = source.metadata.get("rerank_score", 0)
                logger.info(f"  {rank}. VECTOR (score: {score:.4f})")

        # Convert reranked sources back to dicts for storage
        reranked_context = [
            self._clean_context_for_storage(source.model_dump())
            for source in reranked_sources
        ]

        # Create workflow node entry with detailed impact analysis
        execution_order = len(workflow_nodes) + 1
        workflow_node = {
            "node_name": "rerank_results_node",
            "node_type": "tool",
            "results_count": len(reranked_context),
            "summary": f"Reranked {len(vector_sources)} vector sources → {len(reranked_vector)} selected. Kept all {len(graph_sources)} graph sources. Total: {len(reranked_context)} sources.",
            "execution_order": execution_order,
            "details": {
                "vector_sources_before": len(vector_sources),
                "vector_sources_after": len(reranked_vector),
                "vector_sources_removed": vector_removed,
                "vector_removal_rate_percent": round(vector_removal_rate, 1),
                "graph_sources_kept": len(graph_sources),
                "total_sources_after": len(reranked_context),
                "rerank_model": "bge-reranker-v2-m3",
                "rerank_scores": [
                    source.metadata.get("rerank_score", 0)
                    for source in reranked_vector
                ],
                "avg_rerank_score": round(sum([s.metadata.get("rerank_score", 0) for s in reranked_vector]) / len(reranked_vector), 4) if reranked_vector else 0,
                "min_rerank_score": round(min([s.metadata.get("rerank_score", 0) for s in reranked_vector]), 4) if reranked_vector else 0,
                "max_rerank_score": round(max([s.metadata.get("rerank_score", 0) for s in reranked_vector]), 4) if reranked_vector else 0
            }
        }
        workflow_nodes.append(workflow_node)

        state["retrieved_context"] = reranked_context
        state["workflow_nodes"] = workflow_nodes

        logger.info("=" * 80)
        logger.info(f"RERANK_RESULTS_NODE COMPLETE - Selected {len(reranked_context)} sources ({len(reranked_vector)} vector + {len(graph_sources)} graph)")
        logger.info("=" * 80)

        return state

    def _extract_key_context(self, ctx: dict) -> str:
        """
        Extract only the most relevant information from context to minimize token usage.

        Args:
            ctx: Context dictionary from retrieved_context

        Returns:
            Cleaned context string with only essential information
        """
        ctx_type = ctx.get("type", "unknown")

        if ctx_type == "vector":
            # For vector context, include chunk ID and text (truncated)
            chunk_id = ctx.get("chunk_id", "unknown")
            text = ctx.get("data", "")

            # Truncate text to 500 chars to avoid excessive context
            if len(text) > 500:
                text = text[:500] + "..."

            return f"[{chunk_id}] {text}"

        elif ctx_type == "graph":
            # For graph context, extract only key fields
            data = ctx.get("data", {})

            # Build minimal representation
            parts = []

            # Include node type
            if "node_type" in data:
                parts.append(f"Type: {data['node_type']}")

            # Include key identifying information
            if "case_number" in data:
                parts.append(f"Case: {data['case_number']}")
            if "name" in data:
                parts.append(f"Name: {data['name']}")
            if "summary" in data:
                summary = data['summary']
                if len(summary) > 300:
                    summary = summary[:300] + "..."
                parts.append(f"Summary: {summary}")
            if "outcome" in data:
                parts.append(f"Outcome: {data['outcome']}")

            # Include relevance score if available
            if "relevance_score" in data:
                parts.append(f"Relevance: {data['relevance_score']:.2f}")

            return " | ".join(parts) if parts else str(data)

        return ""

    def synthesize_answer_node(self, state: GraphState) -> GraphState:
        """
        LLM node: Generate the final answer based on retrieved context

        """
        logger.info("=" * 80)
        logger.info("EXECUTING SYNTHESIZE_ANSWER_NODE")
        logger.info("=" * 80)

        original_query = state["original_query"]
        retrieved_context = state.get("retrieved_context", [])
        chain_of_thought = state.get("chain_of_thought", [])
        workflow_nodes = state.get("workflow_nodes", [])

        # Build cleaned context for synthesis
        # Note: retrieved_context should already be limited to top 5 reranked sources
        context_items = []
        source_types = {"vector": 0, "graph": 0}

        for idx, ctx in enumerate(retrieved_context):
            cleaned = self._extract_key_context(ctx)
            if cleaned:
                ctx_type = ctx.get("type", "unknown")
                rerank_score = ctx.get("metadata", {}).get("rerank_score", "N/A")

                # Include rerank score in the context
                if isinstance(rerank_score, float):
                    context_items.append(f"[{idx+1}] ({ctx_type.upper()}, relevance: {rerank_score:.3f}) {cleaned}")
                else:
                    context_items.append(f"[{idx+1}] ({ctx_type.upper()}) {cleaned}")

                source_types[ctx_type] = source_types.get(ctx_type, 0) + 1

        # Build synthesis prompt with reranked context
        context_text = "**Top 5 Reranked Sources (by relevance):**\n\n"
        for item in context_items:
            context_text += f"• {item}\n"
        context_text += "\n"

        system_prompt = """You are a legal research assistant specializing in Singapore Family Court cases.
Answer the user's query based ONLY on the provided context.
Cite sources using their IDs like (Source: chunk_id).
Be concise and accurate. If the context doesn't contain enough information, say so."""

        user_prompt = f"""Based on the following information:

{context_text}

Answer this query: {original_query}

Provide a concise answer with source citations."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        answer = response.content.strip()

        # Create workflow node entry for synthesis
        execution_order = len(workflow_nodes) + 1
        workflow_node = {
            "node_name": "synthesize_answer_node",
            "node_type": "synthesis",
            "results_count": len(retrieved_context),
            "summary": f"Generated final answer using LLM synthesis. Used {source_types['vector']} vector sources and {source_types['graph']} graph sources (top 5 reranked). Answer length: {len(answer)} characters.",
            "execution_order": execution_order,
            "details": {
                "vector_sources_used": source_types['vector'],
                "graph_sources_used": source_types['graph'],
                "total_sources_used": len(retrieved_context),
                "answer_length": len(answer),
                "context_optimization": "Top 5 reranked sources using Pinecone's reranking API"
            }
        }
        workflow_nodes.append(workflow_node)

        state["response"] = answer
        state["chain_of_thought"] = chain_of_thought  # v3.0: include CoT in state
        state["workflow_nodes"] = workflow_nodes

        logger.info(f"Generated final answer with reranked context ({source_types['vector']} vector + {source_types['graph']} graph sources)")
        logger.info(f"Chain of thought entries: {len(chain_of_thought)}")
        logger.info("=" * 80)
        logger.info("SYNTHESIZE_ANSWER_NODE COMPLETE")
        logger.info("=" * 80)

        return state

    def _deduplicate_graph_results(self, results: List[Source]) -> List[Source]:
        """
        Remove duplicate graph search results and keep highest relevance.

        Handles both semantic search results (with node properties) and
        structured search results (with Cypher query results like 'a.summary', 'c.case_number').

        Args:
            results: List of Source objects from graph search

        Returns:
            Deduplicated list of Source objects
        """
        seen = {}
        skipped_count = 0

        for result in results:
            node_data = result.data

            # Create a unique key based on node type and identifying properties
            # Skip non-dict data (e.g., strings from vector search)
            if not isinstance(node_data, dict):
                logger.debug(f"Skipping non-dict result: {type(node_data)}")
                skipped_count += 1
                continue

            # Try to extract node properties from Cypher results (e.g., 'c.case_number', 'a.summary')
            # or from semantic search results (e.g., 'case_number', 'summary')

            # Look for case_number (direct or prefixed like 'c.case_number')
            case_number = node_data.get('case_number') or node_data.get('c.case_number')
            if case_number:
                key = ('Case', case_number)
            else:
                # Look for statute (name + section)
                statute_name = node_data.get('name') or node_data.get('s.name')
                statute_section = node_data.get('section') or node_data.get('s.section')
                if statute_name and statute_section:
                    key = ('Statute', statute_name, statute_section)
                else:
                    # Look for argument (summary + outcome)
                    arg_summary = node_data.get('summary') or node_data.get('a.summary')
                    arg_outcome = node_data.get('outcome') or node_data.get('a.outcome')
                    if arg_summary and arg_outcome:
                        key = ('Argument', arg_summary[:50])
                    else:
                        # Look for person (name + role)
                        person_name = node_data.get('name') or node_data.get('p.name')
                        person_role = node_data.get('role') or node_data.get('p.role')
                        if person_name and person_role:
                            key = ('Person', person_name)
                        else:
                            # Look for legal principle (name)
                            lp_name = node_data.get('name') or node_data.get('lp.name')
                            if lp_name:
                                key = ('LegalPrinciple', lp_name)
                            else:
                                # Skip if we can't create a proper key
                                logger.debug(f"Skipping result with unrecognized structure: {list(node_data.keys())}")
                                skipped_count += 1
                                continue

            # Keep result with highest similarity score (for semantic results)
            # or keep first result (for structured results)
            if key not in seen:
                seen[key] = result
            else:
                current_score = result.metadata.get('similarity_score', 0)
                existing_score = seen[key].metadata.get('similarity_score', 0)
                # If current has similarity score and it's higher, replace
                if current_score > 0 and current_score > existing_score:
                    seen[key] = result

        if skipped_count > 0:
            logger.warning(f"Deduplicate: Skipped {skipped_count} results due to unrecognized structure")

        return list(seen.values())

    def _extract_nodes_from_single_result(self, result: Source) -> List[Dict[str, Any]]:
        """
        Extract graph nodes from a single Source result.
        Handles both graph search results (structured data) and vector search results (text chunks).

        Args:
            result: A Source object from search results

        Returns:
            List of node dictionaries extracted from this result
        """
        nodes = []
        data = result.data

        # Handle graph search results (structured data)
        if isinstance(data, dict):
            node_type = None
            node_id = None

            if "case_number" in data:
                node_type = "Case"
                node_id = data.get("case_number")
            elif "section" in data and "name" in data:
                node_type = "Statute"
                node_id = f"{data.get('name')}:{data.get('section')}"
            elif "summary" in data and "outcome" in data:
                node_type = "Argument"
                node_id = data.get("summary", "")[:50]
            elif "name" in data and "role" in data:
                node_type = "Person"
                node_id = data.get("name")
            elif "name" in data:
                node_type = "LegalPrinciple"
                node_id = data.get("name")

            if node_type and node_id:
                node_entry = {
                    "type": node_type,
                    "id": node_id,
                    "data": data,
                    "source_type": result.type,
                    "extraction_method": "graph_search"
                }
                nodes.append(node_entry)
                logger.debug(f"Extracted node from graph: {node_type} - {node_id}")

        # Handle vector search results (text chunks)
        elif isinstance(data, str) and result.type == "vector":
            # Extract case information from metadata
            metadata = result.metadata or {}
            case_extract_name = metadata.get("case_extract_name")
            chunk_id = result.chunk_id

            if case_extract_name:
                # Extract case number from case_extract_name (e.g., "case_extract_2025_21" -> "2025_21")
                case_num = case_extract_name.replace("case_extract_", "")

                node_entry = {
                    "type": "Case",
                    "id": case_num,
                    "data": {
                        "case_extract_name": case_extract_name,
                        "chunk_id": chunk_id,
                        "text_snippet": data[:200] + "..." if len(data) > 200 else data,
                        "similarity_score": metadata.get("score", 0)
                    },
                    "source_type": result.type,
                    "extraction_method": "vector_search"
                }
                nodes.append(node_entry)
                logger.debug(f"Extracted node from vector: Case - {case_num}")

        return nodes





    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent on a query

        Args:
            query: User's question

        Returns:
            Dictionary with 'answer', 'sources', 'workflow_nodes', and 'chain_of_thought'
        """
        logger.info(f"Running agent for query: {query[:100]}...")

        # Initialize state
        initial_state: GraphState = {
            "original_query": query,
            "current_query": None,
            "chain_of_thought": [],  # v3.0: new field
            "strategy": None,  # v3.0: new field
            "retrieved_context": [],
            "plan": None,
            "response": None,
            "needs_more_info": True,
            "iteration_count": 0,
            "workflow_nodes": [],
            "intermediate_steps": [],
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": 100})

        # Extract results
        answer = final_state.get("response", "I couldn't generate an answer.")
        retrieved_context = final_state.get("retrieved_context", [])
        workflow_nodes = final_state.get("workflow_nodes", [])
        chain_of_thought = final_state.get("chain_of_thought", [])  # v3.0: extract CoT

        # Convert context to Source objects
        sources = [Source(**ctx) for ctx in retrieved_context]

        logger.info(f"Agent completed with {len(sources)} sources, {len(workflow_nodes)} workflow nodes, and {len(chain_of_thought)} CoT entries")

        return {
            "answer": answer,
            "sources": sources,
            "workflow_nodes": workflow_nodes,
            "chain_of_thought": chain_of_thought,  # v3.0: include CoT in response
        }


# Singleton instance
_agent = None


def get_agent() -> LegalRAGAgent:
    """Get or create the singleton LegalRAGAgent instance."""
    global _agent
    if _agent is None:
        _agent = LegalRAGAgent()
    return _agent

