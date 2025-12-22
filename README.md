# RAG (Retrieval Augmented Generation) — Example OOP Project 🔧

This repository contains a simple OOP-structured RAG implementation based on code from `use_model_huggingface.ipynb`.

## Structure

- `rag/` - package; contains modules:
  - `loader.py` — PDF loading & chunking
  - `embeddings.py` — embedding model wrapper
  - `vectorstore.py` — FAISS wrapper
  - `llm.py` — LLM wrapper using Hugging Face pipeline
  - `rag.py` — high-level orchestrator that combines the pieces
- `app.py` — CLI for building an index and asking questions
- `requirements.txt` — Python packages used

## Quickstart

1. (Optional) Create and activate a virtualenv
2. Install requirements: `pip install -r requirements.txt`
3. Put PDFs in the `data/` folder
4. Build the index:

```bash
python app.py build-index data --vectorstore vectorstores
```

5. Query the index:

```bash
python app.py query "What datasets are mentioned in the document?"
```

## Notes
- Change default model names in `app.py` or pass them on command line.
- The code uses 4-bit NF4 quantization for the LLM by default; change in `LLMModel` if not needed.

## Web UI (FastAPI)
You can run a small web server to interactively query the RAG model:

```bash
uvicorn rag.api:app --reload --port 8000
```

Serve the UI and API together. Run the server:

```bash
uvicorn rag.api:app --reload --port 8000
```

Then open `http://localhost:8000/` in your browser to access the web UI. The UI will call `/build-index` to start an index build and POST to `/query` with JSON: `{ "query": "...", "k": 5 }` to get answers (response JSON contains `answer`).

