"""
Main FastAPI application for the Legal Agentic RAG system.
Provides the /query endpoint for answering legal questions.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.config import config
from src.models.schemas import QueryRequest, QueryResponse
from src.api.agent import get_agent

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Legal Agentic RAG API")
    logger.info(f"Environment: {config.ENVIRONMENT}")
    
    # Initialize agent (this will initialize all tools)
    try:
        agent = get_agent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Legal Agentic RAG API")


# Create FastAPI app
app = FastAPI(
    title="Legal Agentic RAG API",
    description="Singapore Supreme Family Court Agentic Search System",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "service": "Legal Agentic RAG API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": config.ENVIRONMENT
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Main query endpoint for answering legal questions.
    
    Args:
        request: QueryRequest containing the user's query
        
    Returns:
        QueryResponse with answer and sources
    """
    try:
        logger.info(f"Received query: {request.query[:100]}...")
        
        # Get the agent
        agent = get_agent()
        
        # Run the agent
        result = agent.run(request.query)
        
        # Create response
        response = QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            workflow_nodes=result.get("workflow_nodes", [])
        )
        
        logger.info(f"Query completed successfully with {len(response.sources)} sources")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn for local development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    )

