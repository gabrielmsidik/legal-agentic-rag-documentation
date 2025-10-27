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
from src.tools import get_vector_search_tool, get_graph_search_tool

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
        
        # Build the state graph
        self.graph = self._build_graph()
        logger.info("Initialized LegalRAGAgent")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        # Create the state graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("plan_step", self.plan_step)
        workflow.add_node("vector_search_node", self.vector_search_node)
        workflow.add_node("graph_search_node", self.graph_search_node)
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
                "synthesize": "synthesize_answer_node",
            }
        )
        
        # Add edges from tool nodes to evaluation
        workflow.add_edge("vector_search_node", "evaluate_results_node")
        workflow.add_edge("graph_search_node", "evaluate_results_node")
        
        # Add conditional edges from evaluation
        workflow.add_conditional_edges(
            "evaluate_results_node",
            self.should_continue,
            {
                "continue": "plan_step",
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
        logger.info("Executing plan_step")
        
        original_query = state["original_query"]
        intermediate_steps = state.get("intermediate_steps", [])
        retrieved_context = state.get("retrieved_context", [])
        iteration_count = state.get("iteration_count", 0)
        
        # Build context summary
        context_summary = f"Retrieved {len(retrieved_context)} sources so far."
        if intermediate_steps:
            context_summary += f"\nPrevious actions: {[step[0] for step in intermediate_steps[-3:]]}"
        
        # Create planning prompt
        system_prompt = """You are a planning agent for a legal research system.
            Your job is to decide what action to take next to answer the user's query.

            Available actions:
            1. "vector_search" - Search for relevant text chunks using semantic similarity (best for nuanced, semantic queries)
            2. "graph_search" - Query the knowledge graph for structured information about cases, arguments, statutes, etc. (best for entity-based queries)
            3. "synthesize" - Generate the final answer based on retrieved context

            Strategy:
            - Use BOTH vector and graph search for comprehensive results
            - If no searches have been done yet, start with vector_search for semantic understanding
            - After vector search, use graph_search for structured entity relationships
            - Only synthesize after trying both search types
            - Synthesize when you have diverse results from both search methods

            Respond with ONLY one of: "vector_search", "graph_search", or "synthesize"
            """
        
        user_prompt = f"""Original Query: {original_query}

            {context_summary}

        What action should we take next?"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        plan = response.content.strip().lower()
        
        # Ensure plan is valid
        if "vector" in plan:
            plan = "vector_search"
        elif "graph" in plan:
            plan = "graph_search"
        elif "synth" in plan:
            plan = "synthesize"
        else:
            # Default to synthesize if unclear
            plan = "synthesize"
        
        logger.info(f"Plan decided: {plan}")
        
        state["plan"] = plan
        state["iteration_count"] = iteration_count + 1
        
        return state
    
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
        Tool node: Execute vector search.
        """
        logger.info("Executing vector_search_node")

        original_query = state["original_query"]
        retrieved_context = state.get("retrieved_context", [])
        intermediate_steps = state.get("intermediate_steps", [])
        workflow_nodes = state.get("workflow_nodes", [])

        # Perform vector search
        results = self.vector_tool.search(
            query=original_query,
            top_k=config.VECTOR_SEARCH_TOP_K
        )

        # Add cleaned results to context
        for result in results:
            cleaned_ctx = self._clean_context_for_storage(result.model_dump())
            retrieved_context.append(cleaned_ctx)

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

        # Log the step
        intermediate_steps.append(("vector_search", len(results)))

        state["retrieved_context"] = retrieved_context
        state["intermediate_steps"] = intermediate_steps
        state["workflow_nodes"] = workflow_nodes

        logger.info(f"Vector search returned {len(results)} results")

        return state
    
    def graph_search_node(self, state: GraphState) -> GraphState:
        """
        Tool node: Execute hybrid graph search (semantic + structured).

        Performs both semantic search (using embeddings) and structured search
        (using relationships and keywords) to provide comprehensive results.
        """
        logger.info("Executing graph_search_node with hybrid search")

        original_query = state["original_query"]
        retrieved_context = state.get("retrieved_context", [])
        intermediate_steps = state.get("intermediate_steps", [])
        workflow_nodes = state.get("workflow_nodes", [])

        # Perform semantic search first (using embeddings)
        logger.info("Performing semantic search on graph embeddings...")
        semantic_results = self.graph_tool.semantic_graph_search(query=original_query, top_k=5)
        logger.info(f"Semantic search returned {len(semantic_results)} results")

        # Perform structured search (using relationships and keywords)
        logger.info("Performing structured search on graph relationships...")
        structured_results = self.graph_tool.natural_language_search(query=original_query)
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

        # Add cleaned results to context
        chunk_ids_to_fetch = []
        for result in deduplicated_results:
            cleaned_ctx = self._clean_context_for_storage(result.model_dump())
            retrieved_context.append(cleaned_ctx)
            # Check if result contains vector_chunk_ids
            if result.metadata.get("vector_chunk_ids"):
                chunk_ids_to_fetch.extend(result.metadata["vector_chunk_ids"])

        # If we found chunk IDs, fetch them
        vector_chunks_fetched = 0
        if chunk_ids_to_fetch:
            logger.info(f"Fetching {len(chunk_ids_to_fetch)} vector chunks from graph results")
            vector_results = self.vector_tool.fetch_by_ids(chunk_ids_to_fetch)
            vector_chunks_fetched = len(vector_results)
            for result in vector_results:
                cleaned_ctx = self._clean_context_for_storage(result.model_dump())
                retrieved_context.append(cleaned_ctx)

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

        # Log the step
        intermediate_steps.append(("graph_search_hybrid", len(deduplicated_results)))

        state["retrieved_context"] = retrieved_context
        state["intermediate_steps"] = intermediate_steps
        state["workflow_nodes"] = workflow_nodes

        logger.info(f"Hybrid graph search returned {len(deduplicated_results)} results")

        return state

    def evaluate_results_node(self, state: GraphState) -> GraphState:
        """
        LLM node: Assess if current context is sufficient to answer the query.

        Improved evaluation logic:
        1. Requires both vector and graph search before synthesizing (for diversity)
        2. Checks relevance of results to the query
        3. Only synthesizes when we have sufficient RELEVANT results
        """
        logger.info("Executing evaluate_results_node")

        original_query = state["original_query"]
        retrieved_context = state.get("retrieved_context", [])
        iteration_count = state.get("iteration_count", 0)
        intermediate_steps = state.get("intermediate_steps", [])
        workflow_nodes = state.get("workflow_nodes", [])

        # Track evaluation decision
        evaluation_reason = ""
        decision_made = ""

        # Check iteration limit
        if iteration_count >= config.MAX_ITERATIONS:
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
        has_vector = "vector_search" in search_types
        has_graph = "graph_search" in search_types

        logger.info(f"Search types tried: vector={has_vector}, graph={has_graph}")
        logger.info(f"Retrieved {len(retrieved_context)} sources total")

        # IMPORTANT: Require both search types for diversity
        # If we only have one search type, try the other one
        if not (has_vector and has_graph):
            state["needs_more_info"] = True
            evaluation_reason = f"Need both search types for diversity. Have: vector={has_vector}, graph={has_graph}"
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
                    "has_vector": has_vector,
                    "has_graph": has_graph,
                    "iteration_count": iteration_count
                }
            }
            workflow_nodes.append(workflow_node)
            state["workflow_nodes"] = workflow_nodes
            return state

        # Now that we have both search types, evaluate relevance
        # Build cleaned context summary (minimal tokens)
        context_summary = f"We have {len(retrieved_context)} sources from both vector and graph search:\n\n"
        for idx, ctx in enumerate(retrieved_context[:5]):  # Show only first 5 (was 8)
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
                "has_vector": has_vector,
                "has_graph": has_graph,
                "iteration_count": iteration_count,
                "llm_decision": decision
            }
        }
        workflow_nodes.append(workflow_node)
        state["workflow_nodes"] = workflow_nodes

        return state

    def should_continue(self, state: GraphState) -> Literal["continue", "synthesize"]:
        """
        Conditional logic node: Decide whether to continue searching or synthesize.
        """
        needs_more = state.get("needs_more_info", False)
        return "continue" if needs_more else "synthesize"

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
        LLM node: Generate the final answer based on retrieved context.
        Optimized to minimize token usage by cleaning up context.
        """
        logger.info("Executing synthesize_answer_node")

        original_query = state["original_query"]
        retrieved_context = state.get("retrieved_context", [])
        workflow_nodes = state.get("workflow_nodes", [])

        # Build cleaned context for synthesis
        vector_context = []
        graph_context = []

        for ctx in retrieved_context:
            cleaned = self._extract_key_context(ctx)
            if cleaned:
                if ctx.get("type") == "vector":
                    vector_context.append(cleaned)
                elif ctx.get("type") == "graph":
                    graph_context.append(cleaned)

        # Build synthesis prompt with cleaned context
        context_text = ""

        if vector_context:
            context_text += "**Vector Context (from case documents):**\n"
            # Limit to top 5 most relevant vector sources
            for vc in vector_context[:5]:
                context_text += f"• {vc}\n"
            if len(vector_context) > 5:
                context_text += f"• ... and {len(vector_context) - 5} more sources\n"
            context_text += "\n"

        if graph_context:
            context_text += "**Graph Context (from structured data):**\n"
            # Limit to top 5 most relevant graph sources
            for gc in graph_context[:5]:
                context_text += f"• {gc}\n"
            if len(graph_context) > 5:
                context_text += f"• ... and {len(graph_context) - 5} more sources\n"
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
            "summary": f"Generated final answer using LLM synthesis. Used {len(vector_context)} vector sources and {len(graph_context)} graph sources. Answer length: {len(answer)} characters.",
            "execution_order": execution_order,
            "details": {
                "vector_sources_used": len(vector_context),
                "graph_sources_used": len(graph_context),
                "total_sources_used": len(retrieved_context),
                "answer_length": len(answer),
                "context_optimization": "Cleaned and truncated to minimize token usage"
            }
        }
        workflow_nodes.append(workflow_node)

        state["response"] = answer
        state["workflow_nodes"] = workflow_nodes

        logger.info(f"Generated final answer with cleaned context ({len(vector_context)} vector + {len(graph_context)} graph sources)")

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
        Run the agent on a query.

        Args:
            query: User's question

        Returns:
            Dictionary with 'answer', 'sources', and 'workflow_nodes'
        """
        logger.info(f"Running agent for query: {query[:100]}...")

        # Initialize state
        initial_state: GraphState = {
            "original_query": query,
            "intermediate_steps": [],
            "retrieved_context": [],
            "plan": None,
            "response": None,
            "needs_more_info": True,
            "iteration_count": 0,
            "workflow_nodes": [],
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Extract results
        answer = final_state.get("response", "I couldn't generate an answer.")
        retrieved_context = final_state.get("retrieved_context", [])
        workflow_nodes = final_state.get("workflow_nodes", [])

        # Convert context to Source objects
        sources = [Source(**ctx) for ctx in retrieved_context]

        logger.info(f"Agent completed with {len(sources)} sources and {len(workflow_nodes)} workflow nodes")

        return {
            "answer": answer,
            "sources": sources,
            "workflow_nodes": workflow_nodes
        }


# Singleton instance
_agent = None


def get_agent() -> LegalRAGAgent:
    """Get or create the singleton LegalRAGAgent instance."""
    global _agent
    if _agent is None:
        _agent = LegalRAGAgent()
    return _agent

