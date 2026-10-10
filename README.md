# Neri

## AI-Powered Maintenance Troubleshooting Assistant

Neri is a Retrieval-Augmented Generation (RAG) application designed to help manufacturing floor technicians troubleshoot machine problems using approved technical documentation.

Instead of relying only on general AI knowledge, Neri retrieves relevant information from authorized machine manuals, maintenance logs, and safety procedures before generating a troubleshooting response.

> **Neri helps technicians find the right information faster — it does not control machines or perform repairs autonomously.**

---

## Problem Statement

Maintenance teams often have access to machine manuals, maintenance records, and safety procedures, but finding the correct information during a machine problem can take valuable time.

Neri connects these sources into a single troubleshooting workflow so technicians can:

- Describe a machine problem in natural language.
- Retrieve relevant approved documentation.
- Identify possible causes.
- Follow grounded troubleshooting steps.
- View the sources used to generate the answer.
- See relevant safety warnings.
- Review previous troubleshooting sessions.

---

## Objectives

Neri is designed to:

- Reduce time spent searching maintenance documentation.
- Provide grounded troubleshooting guidance.
- Connect machine problems with relevant maintenance information.
- Surface safety information before troubleshooting.
- Provide traceable source references.
- Avoid generating unsupported maintenance procedures.
- Maintain a history of troubleshooting sessions.

---

## Key Features

### 1. AI Troubleshooting

Technicians provide:

- Machine name
- Machine ID
- Problem description
- Optional error code

Neri retrieves relevant documentation and generates a structured troubleshooting response.

### 2. Retrieval-Augmented Generation

Neri follows a RAG pipeline:

    User Problem
         |
         v
    Query Processing
         |
         v
    Gemini Embedding
         |
         v
    ChromaDB Retrieval
         |
         v
    Relevant Documentation
         |
         v
    Grounded LLM Generation
         |
         v
    Troubleshooting Response

### 3. Source References

Generated answers are accompanied by source information such as:

- Document name
- Document type
- Section
- Page number

This allows technicians to verify the information against the original documentation.

### 4. Safety Guidance

Neri supports safety documentation as a dedicated document type.

Safety-related information is treated separately from ordinary maintenance information to reduce the risk of unsupported safety instructions.

### 5. No-Reliable-Answer Handling

When relevant approved documentation cannot be found, Neri does not intentionally invent an answer.

Instead, it returns a message indicating that there is not enough approved information to provide a reliable answer.

### 6. Troubleshooting History

Previous troubleshooting sessions can be viewed through the History section.

Each session stores:

- Machine
- Machine ID
- Problem
- Error code
- Generated response
- Feedback
- Timestamp

### 7. Feedback

Users can mark troubleshooting responses as:

- Helpful
- Not helpful

Optional feedback comments can also be recorded.

### 8. Document Management

Authorized users can upload approved documentation.

Supported document types:

- Manual
- Maintenance Log
- Safety

Supported file formats:

- PDF
- TXT

Document metadata includes:

- Machine
- Version
- Owner
- Document type
- Processing status
- Number of indexed chunks

### 9. Role-Based Access

Neri currently supports two application roles.

#### Technician

Can:

- Troubleshoot machines
- View troubleshooting history
- Submit feedback

#### Supervisor

Can:

- Troubleshoot machines
- View troubleshooting history
- Submit feedback
- Upload approved documents
- View document information

---

## Technology Stack

### Frontend

- Streamlit

### Backend

- Python
- FastAPI
- Uvicorn

### AI

- Google Gemini
- OpenAI-compatible Gemini API
- Gemini Embeddings

### RAG and Vector Search

- ChromaDB
- Vector embeddings
- Similarity-based retrieval

### Document Processing

- PyPDF
- BeautifulSoup
- Python text processing

### Database

- SQLite

SQLite is used for application data such as troubleshooting history, feedback, and document metadata.

### Utilities

- python-dotenv
- tiktoken
- requests

---

## System Architecture

    +----------------------+
    |     Streamlit UI     |
    |      Neri Frontend   |
    +----------+-----------+
               |
               v
    +----------------------+
    |      FastAPI API     |
    +----------+-----------+
               |
               v
    +----------------------+
    |     RAG Pipeline     |
    |                      |
    | Query                |
    | Retrieval            |
    | Generation           |
    +----+------------+----+
         |            |
         v            v
    +-------------+  +-------------+
    |  ChromaDB   |  |  Gemini LLM |
    | Vector Store|  |  Generation |
    +------+------+  +-------------+
           |
           v
    +----------------------+
    | Approved Documents   |
    |                      |
    | Manuals              |
    | Maintenance Logs     |
    | Safety Procedures    |
    +----------------------+

    +----------------+
    |     SQLite     |
    |                |
    | History        |
    | Feedback       |
    | Documents      |
    +----------------+

---

## RAG Pipeline

Neri uses the following retrieval workflow.

### 1. Document Upload

An authorized user uploads a PDF or TXT document.

The document can be classified as:

- Manual
- Maintenance Log
- Safety

Additional metadata can include:

- Machine
- Version
- Owner

### 2. Document Extraction

Text is extracted from the document.

For PDFs, page information is retained where available.

### 3. Text Cleaning

Extracted text is cleaned before indexing.

### 4. Chunking

Large documents are divided into smaller chunks so relevant sections can be retrieved independently.

### 5. Embedding

Each chunk is converted into a vector embedding using Gemini Embeddings.

### 6. Vector Storage

Embeddings and their metadata are stored in ChromaDB.

### 7. Query Embedding

When a technician submits a problem, the query is converted into an embedding.

### 8. Retrieval

ChromaDB retrieves the most relevant indexed chunks.

Neri also applies machine-specific filtering so documentation assigned to another machine is not used for the troubleshooting request.

### 9. Grounded Generation

The retrieved documentation is passed to Gemini as context.

The model is instructed to use the supplied documentation rather than inventing procedures.

### 10. Source Mapping

The retrieved chunks are converted into source references that are returned with the response.

---

## Safety and Hallucination Guardrails

Neri is designed for a maintenance environment where incorrect instructions can create operational or safety risks.

The system therefore follows several rules.

### Approved Documentation

The generated troubleshooting response is based on retrieved approved documentation.

### No Invented Procedures

Neri is instructed not to create maintenance or safety procedures that are not supported by the retrieved documentation.

### Safety Evidence

Safety-related information should be supported by safety documentation.

### No Autonomous Machine Control

Neri does not:

- Control machines
- Send machine commands
- Perform repairs
- Operate equipment
- Monitor machines in real time

### No Reliable Answer

If sufficient documentation cannot be found, Neri can refuse to provide a troubleshooting answer instead of presenting unsupported information as fact.

---

## Project Structure

    S86-Ai-application_with_RAG-Neri/
    |
    +-- api/
    |   +-- main.py
    |
    +-- app/
    |   +-- streamlit_app.py
    |
    +-- src/
    |   +-- __init__.py
    |   +-- config.py
    |   +-- ingestion.py
    |   +-- embeddings.py
    |   +-- vector_store.py
    |   +-- retrieval.py
    |   +-- llm.py
    |   +-- citations.py
    |   +-- guardrails.py
    |   +-- pipeline.py
    |   +-- cache.py
    |   +-- history.py
    |
    +-- data/
    |   +-- manuals/
    |   +-- maintenance_logs/
    |   +-- safety/
    |
    +-- uploads/
    |
    +-- storage/
    |   +-- chroma_db/
    |
    +-- .env
    +-- .env.example
    +-- .gitignore
    +-- requirements.txt
    +-- README.md
    +-- run.py

---

## Installation

### 1. Clone the Repository

    git clone https://github.com/kalviumcommunity/S86-Ai-application_with_RAG-Neri.git

Move into the project:

    cd S86-Ai-application_with_RAG-Neri

### 2. Create a Virtual Environment

#### Windows

    python -m venv .venv

Activate it:

    .venv\Scripts\activate

#### macOS / Linux

    python3 -m venv .venv

Activate it:

    source .venv/bin/activate

### 3. Install Dependencies

    pip install -r requirements.txt

---

## Environment Configuration

Create a `.env` file in the project root.

Example:

    OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    OPENAI_API_KEY=YOUR_API_KEY

    CHAT_MODEL=gemini-3.1-flash-lite
    EMBED_MODEL=gemini-embedding-001

    VECTOR_DIR=storage/chroma_db
    COLLECTION_NAME=rag_chunks

    RAG_API_URL=http://127.0.0.1:8000

    CACHE_TTL_SECONDS=900

    MODEL_INPUT_COST_PER_1K=0.00015
    MODEL_OUTPUT_COST_PER_1K=0.00060

Never commit the actual API key.

The `.env` file is excluded through `.gitignore`.

---

## Running Neri

The complete application can be started with:

    python run.py

This starts the FastAPI backend and Streamlit frontend.

### FastAPI

    http://127.0.0.1:8000

### Streamlit

Streamlit displays the local application URL in the terminal and normally opens the application in the browser.

---

## API Endpoints

### Health Check

    GET /health

Response:

    {
      "status": "healthy"
    }

### Troubleshooting

    POST /query

Example request:

    {
      "machine": "CNC Mill",
      "machine_id": "CNC-001",
      "problem": "The machine is overheating",
      "error_code": "E102"
    }

### Troubleshooting History

    GET /history

Returns recent troubleshooting sessions.

### History Detail

    GET /history/{session_id}

Returns the complete troubleshooting session.

### Feedback

    POST /history/{session_id}/feedback

Example:

    {
      "feedback": "helpful",
      "comment": "The troubleshooting steps were useful."
    }

### Document Upload

    POST /documents/upload

Supported formats:

- PDF
- TXT

Supported metadata:

- Document type
- Machine
- Version
- Owner

### Documents

    GET /documents

Returns indexed document metadata.

---

## Example Troubleshooting Flow

A technician enters:

    Machine:
    CNC Mill

    Machine ID:
    CNC-001

    Problem:
    Machine is overheating

    Error Code:
    E102

Neri then:

    1. Creates the search query
            |
            v
    2. Generates the query embedding
            |
            v
    3. Searches ChromaDB
            |
            v
    4. Retrieves relevant documentation
            |
            v
    5. Filters documentation for the selected machine
            |
            v
    6. Checks whether usable evidence exists
            |
            v
    7. Sends retrieved context to Gemini
            |
            v
    8. Generates structured troubleshooting guidance
            |
            v
    9. Adds source references
            |
            v
    10. Saves the session to history

---

## Response Structure

A successful troubleshooting response can contain:

    Possible Causes
           |
           v
    Safety Warning
           |
           v
    Troubleshooting Steps
           |
           v
    Source References

If sufficient evidence is not available:

    No Reliable Answer
           |
           v
    We couldn't find enough information in
    approved documentation to provide a reliable answer.

---

## Current Scope

Neri currently focuses on documentation-grounded troubleshooting.

### Included

- Machine troubleshooting
- RAG-based document retrieval
- Manual search
- Maintenance log search
- Safety documentation
- Source references
- Troubleshooting history
- User feedback
- Document upload
- Document metadata
- Role-based access
- Response caching
- FastAPI backend
- Streamlit interface

### Not Included

Neri does not currently provide:

- Machine control
- Autonomous repair
- Predictive maintenance
- IoT monitoring
- Real-time machine monitoring
- Spare-parts ordering
- Voice interface
- Mobile application

---

## Future Scope

Potential future improvements include:

- More advanced retrieval and reranking
- Better document version management
- More detailed machine metadata
- Improved evaluation datasets
- Automated retrieval-quality evaluation
- Authentication integrated with an organization's identity system
- More granular permissions
- Multilingual troubleshooting
- OCR support for scanned manuals
- Integration with maintenance management systems
- Analytics for recurring machine problems

---

## Design Principles

### Grounded

Responses should be based on approved documentation.

### Traceable

Users should be able to identify the sources used for an answer.

### Safe

Safety-related guidance should not be generated without supporting evidence.

### Human-in-the-loop

Neri assists technicians rather than replacing their judgment.

### Controlled

Neri does not directly control machines or perform physical actions.

---

## Development

The application is organized into separate layers.

### Frontend

The `app/` directory contains the Streamlit user interface.

### API

The `api/` directory contains FastAPI endpoints.

### RAG Logic

The `src/` directory contains:

- Document ingestion
- Embeddings
- Vector storage
- Retrieval
- LLM generation
- Citations
- Guardrails
- Pipeline
- Caching
- History management

### Data

The `data/` directory contains source documentation organized into:

- Manuals
- Maintenance logs
- Safety documents

### Storage

The `storage/` directory contains local application databases and the ChromaDB vector store.

---

## Automated Testing Suite

NERI is equipped with an automated test suite comprising **65 tests** organized across three dedicated modules to ensure system reliability, safety guardrail enforcement, and deterministic retrieval:

1. **`tests/test_phase1_auth_and_safety.py` (22 tests)**:
   - Fail-closed API authentication validation (`X-API-Key` & `Authorization: Bearer`).
   - Role-based access control (Technician vs. Supervisor privilege boundaries).
   - Safety guardrail escalation: automatic hazard detection, prompt refusal when safety evidence is absent, and strict citation enforcement.

2. **`tests/test_phase2_retrieval_and_integrity.py` (16 tests)**:
   - Machine ID normalization and canonical slug indexing (e.g., `CNC Mill` -> `cnc-mill`).
   - ChromaDB vector store metadata filtering and cross-machine chunk isolation.
   - Text extraction, chunk boundary integrity, and deduplication during document re-indexing.

3. **`tests/test_rag_pipeline_and_services.py` (27 tests)**:
   - Full RAG troubleshooting pipeline integration with deterministic mocks.
   - Robust LLM error handling: malformed JSON recovery, empty completion fallback, and refusal formatting.
   - Grounded context synthesis, source attribution mapping, and prompt-injection safety defenses.
   - Atomic document deletion consistency across SQLite records and ChromaDB vector embeddings.

### Running the Test Suite

```bash
# Execute the full automated test suite with short traceback
pytest -v --tb=short

# Run a specific test suite
pytest tests/test_rag_pipeline_and_services.py -v
```

**Test Status:** 65 passed, 0 failed (100% pass rate in ~20 seconds).

---

## Continuous Integration & Delivery (CI/CD)

NERI implements a continuous integration workflow via **GitHub Actions** (`.github/workflows/ci.yml`) to ensure every code change maintains security and functional requirements.

### Workflow Stages

1. **Code Integrity & Syntax Check (`lint-and-validate`)**:
   - Automated checkout on Python 3.11.
   - Dependency resolution and verification against `requirements.txt`.
   - Bytecode compilation check across all modules: `python -m compileall api src app tests`.

2. **Automated Unit & RAG Regression Tests (`test-suite`)**:
   - Executes the complete 65-test suite in a clean, isolated environment.
   - Injects mock test API keys to verify fail-closed authorization without exposing production secrets.
   - Validates safety refusal behaviors and retrieval pipelines.

3. **Load Test Harness Verification (`load-test-verification`)**:
   - Compiles both Locust load suites and standalone runner scripts.
   - Validates CLI argument parsing and dry-run execution (`python load_tests/run_load_test.py --help`).

### Workflow Triggers

- Automatic execution on `push` to `main` and `master`.
- Pull request evaluation on `main` and `master`.
- Manual workflow dispatch capability via GitHub Actions UI.

---

## Automated Load & Performance Testing

NERI includes automated load-testing tooling to benchmark response times, throughput (RPS), and security enforcement under concurrent technician load.

### Load Testing Tools

1. **Locust Simulation (`load_tests/locustfile.py`)**:
   - Simulates field technicians querying manuals, viewing history, and submitting diagnostic requests.
   - Simulates monitoring probes tracking `/health` endpoints.
   - Verifies fail-closed authentication rejection under load.

2. **Standalone Runner (`load_tests/run_load_test.py`)**:
   - Zero-external-dashboard Python benchmark harness using multi-threaded execution.
   - Tracks response latencies and computes percentiles (**p50, p90, p95, p99**), error rates, and throughput.
   - Enforces SLA threshold assertions for CI/CD gates.

### Measured Benchmark Results

Conducted against a live instance with **5 concurrent simulated technician threads** over 52 requests:

| Target Endpoint | Requests | Throughput (RPS) | Avg Latency | p50 (Median) | p95 Latency | p99 Latency | Expected Result / Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **`GET /health`** | 17 | 1.26 req/s | 8.9 ms | 3.3 ms | 26.4 ms | 74.6 ms | 200 OK — Lightweight health check |
| **`GET /documents`** | 10 | 0.74 req/s | 13.1 ms | 10.2 ms | 23.4 ms | 23.4 ms | 200 OK — Document catalog retrieval |
| **`GET /history`** | 13 | 0.96 req/s | 39.4 ms | 11.0 ms | 132.7 ms | 135.8 ms | 200 OK — Session history retrieval |
| **`GET /documents (Unauthorized)`** | 6 | 0.44 req/s | 5.5 ms | 4.8 ms | 10.1 ms | 10.1 ms | 401 Unauthorized — Instant fail-closed |
| **`POST /query` (Live RAG)** | 6 | 0.44 req/s | 8,683.1 ms | 9,665.2 ms | 12,668.6 ms | 12,668.6 ms | 200 OK — ChromaDB retrieval + LLM inference |
| **Total / Aggregate** | **52** | **3.84 req/s** | - | **9.22 ms** | - | - | **0.0% Failure Rate (52/52 Succeeded)** |

### How to Run Load Tests

```bash
# 1. Run the standalone load benchmark (5 concurrent users, 10s duration)
python load_tests/run_load_test.py --host http://127.0.0.1:8000 --users 5 --duration 10

# 2. Run with JSON report export
python load_tests/run_load_test.py --host http://127.0.0.1:8000 --users 5 --duration 10 --output load_tests/benchmark_results.json

# 3. Run dry-run validation (no running server needed)
python load_tests/run_load_test.py --dry-run

# 4. Run headless Locust load test
locust -f load_tests/locustfile.py --headless -u 5 -r 2 --run-time 15s --host http://127.0.0.1:8000

# 5. Run Locust with interactive web UI
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000
# Open http://localhost:8089 in your browser
```

---

## Project Information

**Project Name:** Neri

**Type:** AI Application with Retrieval-Augmented Generation

**Domain:** Industrial Maintenance

**Primary Users:** Manufacturing Floor Technicians

**Supporting Users:**

- Maintenance Supervisors
- Plant Engineers
- Safety Officers

---

## License

This project was developed as an academic/project application.