# Professional RAG System 🚀

This repository implements a versatile **Retrieval-Augmented Generation (RAG)** system with dual-mode capabilities: **Vector Retrieval** (for unstructured documents) and **SQL Agency** (for structured data). Built with an Object-Oriented approach, it's designed for scalability and ease of integration.

## 🌟 Key Features

-   **Dual-Agent System**:
    -   **Vector RAG**: Processes PDFs, chunks text, and uses FAISS for high-performance semantic search.
    -   **SQL Agent**: Translates natural language questions into SQL queries to extract insights from SQLite databases.
-   **Advanced LLM Integration**: Supports Hugging Face models (e.g., Qwen, Phi-2) with 4-bit quantization (NF4) for efficient local execution.
-   **FastAPI Backend**: A robust API to serve queries, build indices, and manage models dynamically.
-   **Modern Web UI**: A user-friendly interface for conversational interaction.
-   **CLI Tooling**: Comprehensive command-line interface for indexing and querying.

## 📂 Project Structure

-   `rag/` - Core package:
    -   `loader.py`: PDF document loading and intelligent chunking.
    -   `embeddings.py`: Wrapper for Sentence Transformers/BGE embedding models.
    -   `vectorstore.py`: FAISS-based vector database management.
    -   `llm.py`: Unified interface for Large Language Models.
    -   `rag.py`: Orchestrator for the Vector RAG workflow.
    -   `agent.py`: LangChain-powered SQL Agent for structured data queries.
    -   `api.py`: FastAPI implementation for web services.
    -   `sql_db.py`: Utilities for managing and converting CSVs to SQLite.
-   `app.py`: Main entry point for CLI operations.
-   `web_ui/`: Frontend assets for the web interface.
-   `data/`: Directory for source documents (PDFs, CSVs).
-   `vectorstores/`: Persistent storage for FAISS indices.

## 🚀 Quickstart

### 1. Prerequisites
Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment:
```powershell
python -m venv .env
.\.env\Scripts\Activate.ps1
```

### 2. Installation
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 3. Usage via CLI

#### Build Vector Index
Place your PDFs in the `data/` folder and run:
```bash
python app.py build-index data --vectorstore vectorstores
```

#### Build SQL Database
Convert CSV data into a queryable SQLite database:
```bash
python app.py build-sql-db data --db-path hospital.db
```

#### Querying
The CLI defaults to the SQL agent if structured keywords are detected, or you can specify the agent:
```bash
# Ask about unstructured document content
python app.py query "Summarize the findings in the documents" --agent-type rag

# Ask about structured data (e.g., in hospital.db)
python app.py query "How many patients were admitted in December?" --agent-type sql
```

## 🌐 Web Interface & API

Start the FastAPI server to access the Web UI and API endpoints:
```bash
uvicorn rag.api:app --reload --port 8000
```
-   **Web UI**: Open `http://localhost:8000/ui/` in your browser.
-   **API Health**: `GET /health`
-   **Query Endpoint**: `POST /query` with payload `{ "query": "..." }`

## ⚙️ Configuration

-   **Models**: Default models are defined in `rag/api.py` and can be overridden via CLI arguments or API calls.
-   **Quantization**: Enabled by default in `LLMModel` for GPU memory efficiency.
-   **Language Support**: Pre-configured with models like `dangvantuan/vietnamese-embedding` for optimized Vietnamese support.

## 🛠️ Requirements
-   `torch`, `transformers`, `bitsandbytes` (for LLM)
-   `langchain`, `faiss-cpu`/`faiss-gpu` (for RAG)
-   `fastapi`, `uvicorn` (for API)

