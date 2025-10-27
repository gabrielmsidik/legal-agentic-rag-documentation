# Singapore Supreme Family Court Agentic Search

A hybrid vector-based + Knowledge Graph based system, built with an LLM agent to answer questions related to Singapore Supreme Family Court cases.

## Architecture

This system combines:
- **Vector Search**: Pinecone with `llama-text-embed-v2` for semantic search
- **Knowledge Graph**: Neo4j Aura for structured legal reasoning
- **LLM Agent**: OpenAI `gpt-4o-mini` with LangGraph for orchestration

## Project Structure

```
legal-agentic-rag-documentation/
├── src/
│   ├── api/              # FastAPI application and LangGraph agent
│   ├── ingestion/        # Graph ingestion pipeline
│   ├── models/           # Pydantic schemas and models
│   ├── tools/            # Vector and graph search tools
│   └── config.py         # Configuration management
├── tests/                # Unit tests
├── main.py               # FastAPI application entry point
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key
- Pinecone API key and index
- Neo4j Aura instance

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd legal-agentic-rag-documentation
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=legal-agentic-rag
PINECONE_NAMESPACE=singapore-family-court

# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
NEO4J_DATABASE=neo4j

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## Usage

### Running the API Server

For development:
```bash
python main.py
```

For production (using Gunicorn):
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --timeout 120
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### POST /query

Query the legal knowledge base:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are cases about custody of children?"}'
```

Response:
```json
{
  "answer": "Based on the retrieved information...",
  "sources": [
    {
      "type": "vector",
      "chunk_id": "case_extract_2024_1_section_5",
      "data": "Text content...",
      "metadata": {...}
    }
  ]
}
```

#### GET /health

Health check endpoint:
```bash
curl http://localhost:8000/health
```

### Running the Graph Ingestion Pipeline

To ingest a case into the Neo4j knowledge graph:

```bash
python -m src.ingestion.graph_ingestion case_extract_2024_1
```

This will:
1. Fetch all chunks for the case from Pinecone
2. Extract entities (Case, Person, Statute, LegalPrinciple, Argument)
3. Extract relationships between entities
4. Store everything in Neo4j

## Testing

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_agent.py
```

Run with coverage:
```bash
pytest --cov=src tests/
```

**Note**: Tests use actual credentials and perform read-only operations (except for small isolated create+delete tests in `test_graph_ingestion.py`).

## Development

### Code Structure

- **`main.py`**: FastAPI application with `/query` endpoint
- **`src/api/agent.py`**: LangGraph agent implementation with state machine
- **`src/tools/vector_search.py`**: Pinecone vector search tool
- **`src/tools/graph_search.py`**: Neo4j graph search with text-to-Cypher
- **`src/ingestion/graph_ingestion.py`**: Graph ingestion pipeline
- **`src/models/schemas.py`**: Pydantic models for API and internal state
- **`src/config.py`**: Configuration management

### LangGraph Agent Flow

The agent uses a state machine with the following nodes:

1. **plan_step**: Decide next action (vector search, graph search, or synthesize)
2. **vector_search_node**: Execute semantic search on Pinecone
3. **graph_search_node**: Execute Cypher query on Neo4j
4. **evaluate_results_node**: Check if we have enough context
5. **synthesize_answer_node**: Generate final answer with citations

## Deployment

### Render Deployment

The application is configured for deployment on Render. See `render.yaml` for configuration.

Build command:
```bash
pip install -r requirements.txt
```

Start command:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --timeout 120
```

Make sure to set all environment variables in the Render dashboard.

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]

