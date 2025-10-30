# Quick Start Guide

## Initial Setup

1. **Create virtual environment**:
```bash
python -m venv venv
```

2. **Activate virtual environment**:
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment**:
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual credentials
# Required: OPENAI_API_KEY, PINECONE_API_KEY, NEO4J_URI, NEO4J_PASSWORD
```

## Running the API

### Development Mode
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Test the API
```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What are cases about custody?\"}"
```

## Running Graph Ingestion

To ingest a case into Neo4j:

```bash
python -m src.ingestion.graph_ingestion case_extract_2024_1
```

Replace `case_extract_2024_1` with your actual case extract name from Pinecone.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agent.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src tests/
```

## Project Structure Overview

```
├── main.py                          # FastAPI application entry point
├── src/
│   ├── config.py                    # Configuration management
│   ├── api/
│   │   └── agent.py                 # LangGraph agent implementation
│   ├── tools/
│   │   ├── vector_search.py         # Pinecone vector search
│   │   └── graph_search.py          # Neo4j graph search
│   ├── models/
│   │   └── schemas.py               # Pydantic models
│   └── ingestion/
│       └── graph_ingestion.py       # Graph ingestion pipeline
└── tests/                           # Unit tests
```

## Common Issues

### Issue: Import errors
**Solution**: Make sure you're in the project root directory and virtual environment is activated.

### Issue: Missing environment variables
**Solution**: Check that your `.env` file exists and contains all required variables from `.env.example`.

### Issue: Connection errors to Pinecone/Neo4j
**Solution**: Verify your API keys and connection strings in the `.env` file.

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Next Steps

1. Ensure your Pinecone index has data (vector chunks from eLitigation cases)
2. Run the graph ingestion pipeline for cases you want to query
3. Test the `/query` endpoint with various legal questions
4. Review the sources returned to understand how the agent retrieves information

