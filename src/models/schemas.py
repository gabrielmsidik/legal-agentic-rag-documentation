"""
Pydantic models for API requests, responses, and internal state.
"""

from typing import List, Union, Dict, Any, TypedDict, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


# Enum Definitions for Categorical Fields

class PersonRole(str, Enum):
    """Enum for Person node roles."""
    JUDGE = "Judge"
    APPELLANT = "Appellant"
    RESPONDENT = "Respondent"
    LAWYER_APPELLANT = "Lawyer for Appellant"
    LAWYER_RESPONDENT = "Lawyer for Respondent"
    WITNESS = "Witness"
    MEDIATOR = "Mediator"
    OTHERS = "Others"


class LegalPrincipleType(str, Enum):
    """Enum for LegalPrinciple node types (case types and legal concepts)."""
    DIVISION_MATRIMONIAL_ASSETS = "Division of Matrimonial Assets"
    CUSTODY_CARE_CONTROL = "Custody, Care and Control"
    MAINTENANCE_SUPPORT = "Maintenance and Support"
    GUARDIANSHIP = "Guardianship"
    ADOPTION = "Adoption"
    ANCILLARY_MATTERS = "Ancillary Matters"
    FAMILY_VIOLENCE = "Family Violence"
    MATRIMONIAL_PROPERTY = "Matrimonial Property"
    CHILD_PROTECTION = "Child Protection"
    SUCCESSION = "Succession"
    OTHERS = "Others"


class ArgumentOutcome(str, Enum):
    """Enum for Argument node outcomes."""
    ACCEPTED = "Accepted"
    REJECTED = "Rejected"
    PARTIALLY_ACCEPTED = "Partially Accepted"
    DISMISSED = "Dismissed"
    ALLOWED = "Allowed"
    DISMISSED_WITH_COSTS = "Dismissed with Costs"
    OTHERS = "Others"


class RelationshipType(str, Enum):
    """Enum for all relationship types in the graph."""
    PRESIDED_OVER = "PRESIDED_OVER"
    IS_PARTY_IN = "IS_PARTY_IN"
    REPRESENTED = "REPRESENTED"
    CITES = "CITES"
    CONTAINS = "CONTAINS"
    IS_RELATED_TO = "IS_RELATED_TO"
    MADE = "MADE"
    IS_ABOUT = "IS_ABOUT"
    REFERENCES = "REFERENCES"


# Relationship Definition Models

class RelationshipDefinition(BaseModel):
    """Base model for relationship definitions."""
    type: RelationshipType = Field(..., description="Type of relationship")
    source_label: str = Field(..., description="Source node label (e.g., 'Person')")
    target_label: str = Field(..., description="Target node label (e.g., 'Case')")
    properties: Dict[str, str] = Field(default_factory=dict, description="Relationship properties and their types")
    description: str = Field(..., description="Human-readable description of the relationship")


# Predefined relationship definitions
RELATIONSHIP_DEFINITIONS = {
    RelationshipType.PRESIDED_OVER: RelationshipDefinition(
        type=RelationshipType.PRESIDED_OVER,
        source_label="Person",
        target_label="Case",
        properties={},
        description="A judge presided over a case"
    ),
    RelationshipType.IS_PARTY_IN: RelationshipDefinition(
        type=RelationshipType.IS_PARTY_IN,
        source_label="Person",
        target_label="Case",
        properties={"objective": "string"},
        description="A person is a party in a case (appellant, respondent, etc.)"
    ),
    RelationshipType.REPRESENTED: RelationshipDefinition(
        type=RelationshipType.REPRESENTED,
        source_label="Person",
        target_label="Person",
        properties={},
        description="A lawyer represented a party"
    ),
    RelationshipType.CITES: RelationshipDefinition(
        type=RelationshipType.CITES,
        source_label="Case",
        target_label="Case",
        properties={},
        description="A case cites another case as precedent"
    ),
    RelationshipType.CONTAINS: RelationshipDefinition(
        type=RelationshipType.CONTAINS,
        source_label="Case",
        target_label="Argument",
        properties={},
        description="A case contains an argument"
    ),
    RelationshipType.IS_RELATED_TO: RelationshipDefinition(
        type=RelationshipType.IS_RELATED_TO,
        source_label="Case",
        target_label="LegalPrinciple",
        properties={},
        description="A case is related to a legal principle"
    ),
    RelationshipType.MADE: RelationshipDefinition(
        type=RelationshipType.MADE,
        source_label="Person",
        target_label="Argument",
        properties={},
        description="A person made an argument"
    ),
    RelationshipType.IS_ABOUT: RelationshipDefinition(
        type=RelationshipType.IS_ABOUT,
        source_label="Argument",
        target_label="LegalPrinciple",
        properties={},
        description="An argument is about a legal principle"
    ),
    RelationshipType.REFERENCES: RelationshipDefinition(
        type=RelationshipType.REFERENCES,
        source_label="Argument",
        target_label="Case",
        properties={},
        description="An argument references another case"
    ),
}


# API Request/Response Models

class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""
    query: str = Field(..., description="The user's question about Singapore Family Court cases")


class Source(BaseModel):
    """Model representing a source of information used to generate an answer."""
    type: str = Field(..., description="Source type: 'vector' or 'graph'")
    chunk_id: Union[str, None] = Field(None, description="ID of the vector chunk from Pinecone")
    data: Union[str, Dict[str, Any]] = Field(..., description="Retrieved text chunk (vector) or structured data (graph)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the source (e.g., case_number, section)")


class WorkflowNode(BaseModel):
    """Model representing a node in the LangGraph workflow execution."""
    node_name: str = Field(..., description="Name of the LangGraph node (e.g., 'vector_search_node', 'graph_search_node')")
    node_type: str = Field(..., description="Type of node: 'tool', 'decision', 'synthesis'")
    results_count: int = Field(..., description="Number of results returned by this node")
    summary: str = Field(..., description="Brief summary of what this node did and its results")
    execution_order: int = Field(..., description="Order in which this node was executed (1-indexed)")


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""
    answer: str = Field(..., description="The final synthesized answer from the LLM")
    sources: List[Source] = Field(..., description="List of sources used to generate the answer")
    workflow_nodes: List[WorkflowNode] = Field(default_factory=list, description="List of LangGraph workflow nodes executed to generate the answer (for explainability and debugging)")


# LangGraph State Models

class GraphState(TypedDict):
    """
    State definition for the LangGraph agent.
    This is passed between nodes in the state machine.
    """
    original_query: str                  # The initial user question
    intermediate_steps: List[tuple]      # History of tool calls and results (for logging/debugging)
    retrieved_context: List[Dict]        # Accumulated context from tools (vector + graph)
    plan: Union[str, None]               # Agent's plan or next action decision
    response: Union[str, None]           # Final generated answer
    needs_more_info: bool                # Flag to control potential looping
    iteration_count: int                 # Track number of iterations to prevent infinite loops
    workflow_nodes: List[Dict[str, Any]] # LangGraph workflow nodes executed (for explainability)


# Tool Input/Output Models

class VectorSearchInput(BaseModel):
    """Input model for vector search tool."""
    query: Union[str, None] = Field(None, description="Semantic search query")
    fetch_ids: Union[List[str], None] = Field(None, description="Specific chunk IDs to fetch")
    top_k: int = Field(5, description="Number of results to return for semantic search")


class GraphSearchInput(BaseModel):
    """Input model for graph search tool."""
    query: str = Field(..., description="Natural language query to convert to Cypher")


# Graph Ingestion Models

class CaseNode(BaseModel):
    """Model for Case node in Neo4j."""
    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    case_number: str = Field(..., description="Unique case identifier (e.g., '2024_SGHCF_1')")
    name: str = Field(..., description="Case name (e.g., 'ABC v XYZ')")
    date: str = Field(..., description="Date of judgment")
    year: int = Field(..., description="Year of judgment")
    summary: str = Field(..., description="High-level background of the case") # Useful for semantic search
    summary_embedding: Optional[List[float]] = Field(None, description="Vector embedding of case summary for semantic search")
    outcome: str = Field(..., description="Final conclusion") # Useful for semantic search
    outcome_embedding: Optional[List[float]] = Field(None, description="Vector embedding of case outcome for semantic search")


class PersonNode(BaseModel):
    """Model for Person node in Neo4j."""
    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    name: str = Field(..., description="Canonical name or composite ID")
    role: Union[PersonRole, str] = Field(..., description="Role (e.g., 'Judge', 'Appellant', 'Respondent')")
    role_others: Optional[str] = Field(None, description="Additional role details if role is 'Others'")


class StatuteNode(BaseModel):
    """Model for Statute node in Neo4j."""
    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    name: str = Field(..., description="Name of the act")
    section: str = Field(..., description="Specific section")
    summary: str = Field(..., description="Summary of the statute") # Useful for semantic search
    summary_embedding: Optional[List[float]] = Field(None, description="Vector embedding of statute summary for semantic search")


class LegalPrincipleNode(BaseModel):
    """Model for LegalPrinciple node in Neo4j."""
    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    name: Union[LegalPrincipleType, str] = Field(..., description="Primary legal concept or case type")
    name_others: Optional[str] = Field(None, description="Additional principle details if name is 'Others'")


class ArgumentNode(BaseModel):
    """Model for Argument node in Neo4j."""
    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    summary: str = Field(..., description="1-2 sentence summary of the party's claim") # Useful for semantic search
    summary_embedding: Optional[List[float]] = Field(None, description="Vector embedding of argument summary for semantic search")
    judge_reasoning: str = Field(..., description="1-2 sentence summary of judge's evaluation") # Useful for semantic search
    judge_reasoning_embedding: Optional[List[float]] = Field(None, description="Vector embedding of judge's reasoning for semantic search")
    outcome: Union[ArgumentOutcome, str] = Field(..., description="Result for this argument (e.g., 'Accepted', 'Rejected')")
    outcome_others: Optional[str] = Field(None, description="Additional outcome details if outcome is 'Others'")
    section: str = Field(..., description="Document section where argument is discussed")
    vector_chunk_ids: List[str] = Field(..., description="List of Pinecone chunk IDs containing full text")


class ExtractionResult(BaseModel):
    """Model for the complete extraction result from a case."""
    case: CaseNode
    persons: List[PersonNode] = Field(default_factory=list)
    statutes: List[StatuteNode] = Field(default_factory=list)
    legal_principles: List[LegalPrincipleNode] = Field(default_factory=list)
    arguments: List[ArgumentNode] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="List of relationship definitions")


# Extraction Models for OpenAI Structured Output API

# Simplified models for OpenAI (without embedding fields)
class CaseNodeForOpenAI(BaseModel):
    """Simplified Case node for OpenAI extraction (no embeddings)."""
    model_config = ConfigDict(
        json_schema_extra={
            "additionalProperties": False,
            "required": ["case_number", "name", "date", "year", "summary", "outcome"]
        }
    )

    case_number: str = Field(..., description="Unique case identifier (e.g., '2024_SGHCF_1')")
    name: str = Field(..., description="Case name (e.g., 'ABC v XYZ')")
    date: str = Field(..., description="Date of judgment")
    year: int = Field(..., description="Year of judgment")
    summary: str = Field(..., description="High-level background of the case")
    outcome: str = Field(..., description="Final conclusion")


class PersonNodeForOpenAI(BaseModel):
    """Simplified Person node for OpenAI extraction."""
    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    name: str = Field(..., description="Canonical name or composite ID")
    role: Union[PersonRole, str] = Field(..., description="Role (e.g., 'Judge', 'Appellant', 'Respondent')")


class LegalPrincipleNodeForOpenAI(BaseModel):
    """Simplified LegalPrinciple node for OpenAI extraction."""
    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    name: Union[LegalPrincipleType, str] = Field(..., description="Primary legal concept or case type")


class StatuteNodeForOpenAI(BaseModel):
    """Simplified Statute node for OpenAI extraction (no embeddings)."""
    model_config = ConfigDict(
        json_schema_extra={
            "additionalProperties": False,
            "required": ["name", "section", "summary"]
        }
    )

    name: str = Field(..., description="Name of the act")
    section: str = Field(..., description="Specific section")
    summary: str = Field(..., description="Summary of the statute")


class ArgumentNodeForOpenAI(BaseModel):
    """Simplified Argument node for OpenAI extraction (no embeddings)."""
    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    summary: str = Field(..., description="1-2 sentence summary of the party's claim")
    judge_reasoning: str = Field(..., description="1-2 sentence summary of judge's evaluation")
    outcome: Union[ArgumentOutcome, str] = Field(..., description="Result for this argument (e.g., 'Accepted', 'Rejected')")
    section: str = Field(..., description="Document section where argument is discussed")
    vector_chunk_ids: List[str] = Field(..., description="List of Pinecone chunk IDs containing full text")


class FirstPassExtractionResult(BaseModel):
    """Structured output model for first pass extraction (Case, Person, LegalPrinciple)."""
    model_config = ConfigDict(
        json_schema_extra={
            "additionalProperties": False,
            "required": ["case", "persons", "legal_principles"]
        }
    )

    case: CaseNodeForOpenAI = Field(..., description="Case information")
    persons: List[PersonNodeForOpenAI] = Field(default_factory=list, description="All persons mentioned in the case")
    legal_principles: List[LegalPrincipleNodeForOpenAI] = Field(default_factory=list, description="All legal principles mentioned")


class SecondPassExtractionResult(BaseModel):
    """Structured output model for second pass extraction (Arguments, Statutes, References)."""
    model_config = ConfigDict(
        json_schema_extra={
            "additionalProperties": False,
            "required": ["arguments", "statutes", "referenced_cases"]
        }
    )

    arguments: List[ArgumentNodeForOpenAI] = Field(default_factory=list, description="All arguments made in the case")
    statutes: List[StatuteNodeForOpenAI] = Field(default_factory=list, description="All statutes referenced")
    referenced_cases: List[str] = Field(default_factory=list, description="List of case numbers referenced")

