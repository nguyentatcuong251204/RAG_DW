from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
from rag import DocumentLoader, EmbeddingModel, VectorStore, LLMModel, RAG, embeddings
from rag.agent import SQLAgent
import threading
import os

app = FastAPI(title="RAG Chat API")


if os.path.isdir("web_ui"):
    app.mount("/ui", StaticFiles(directory="web_ui", html=True), name="web_ui")

# enable permissive CORS (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (lazy init)
_components = {
    "loader": None,
    "emb": None,
    "vs": None,
    "llm": None,
    
    "agent_sql": None,
    "building": False,
}

class BuildIndexResponse(BaseModel):
    status: str
    message: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str

class ReloadModelRequest(BaseModel):
    model_name: Optional[str] = None
    max_new_tokens: Optional[int] = None

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# embeddings_id = "BAAI/bge-base-en-v1.5"


# model_id = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
embeddings_id = "dangvantuan/vietnamese-embedding"

# Initialize and return shared components. Accepts optional embedding model and vector store directory.
# If reload=True the function will clear and re-initialize components that depend on the LLM.
def ensure_components(llm_model: str = model_id, embedding_model: str = embeddings_id, store_dir: str = "vectorstores", reload: bool = False, max_new_tokens: int = 120):
    # Optionally clear LLM & dependent components so a fresh model can be loaded on demand
    if reload:
        if _components.get("llm") is not None:
            _components["llm"] = None
            _components["agent_sql"] = None
            _components.pop("rag", None)

    # Embedding model
    if _components["emb"] is None:
        _components["emb"] = EmbeddingModel(model_name=embedding_model)

    # Vector store
    if _components["vs"] is None:
        # VectorStore expects (embedding_model, store_dir)
        _components["vs"] = VectorStore(_components["emb"], store_dir=store_dir)

    # Attempt to load vectorstore from disk if present to avoid lazy-load errors later
    try:
        if _components["vs"] and _components["vs"].db is None:
            if os.path.isdir(store_dir) and len(os.listdir(store_dir)) > 0:
                try:
                    _components["vs"].load()
                    print("VectorStore loaded from disk on startup")
                except Exception as e:
                    print(f"VectorStore load failed: {e}")
    except Exception:
        pass

    # Document loader
    if _components["loader"] is None:
        _components["loader"] = DocumentLoader()

    # LLM (only instantiate if missing)
    if _components["llm"] is None:
        _components["llm"] = LLMModel(model_name=llm_model, max_new_tokens=max_new_tokens)

    # SQL agent (depends on LLM)
    if _components["agent_sql"] is None:
        _components["agent_sql"] = SQLAgent(db_path="hospital.db", llm_model=_components["llm"])

    # RAG (optional)
    if _components.get("rag") is None:
        # RAG expects (loader, embedding_model, vector_store, llm)
        _components["rag"] = RAG(
            loader=_components["loader"],
            embedding_model=_components["emb"],
            vector_store=_components["vs"],
            llm=_components["llm"],
        )

    return _components


def is_structured_query(text: str) -> bool:
    """Heuristic to determine whether a user question should be answered by the SQL agent.
    Returns True for queries that contain aggregation or SQL-like keywords (counts, sums, select, where, group by, etc.).
    """
    if not text:
        return False
    q = text.lower()
    keywords = ["how many","sum","average","count","total","mean","max","min","select","sql","rows","columns","join","group by","order by","where","filter","top","find","list","count of","number of"]
    return any(k in q for k in keywords)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    # Redirect to the UI if present
    if os.path.isdir("web_ui"):
        return RedirectResponse(url="/ui/")
    return {"status": "ok"} 


@app.post("/build-index", response_model=BuildIndexResponse)
async def build_index(pdf_dir: str = "data", vectorstore: str = "vectorstores", chunk_size: int = 215, chunk_overlap: int = 50, embedding_model: str = embeddings_id):
    # Build off-thread to avoid blocking long-running work in the API process
    try:
        comps = ensure_components(embedding_model=embedding_model, store_dir=vectorstore)
        loader: DocumentLoader = comps["loader"]
        emb: EmbeddingModel = comps["emb"]
        vs: VectorStore = comps["vs"]

        # allow custom chunk sizes
        loader.chunk_size = chunk_size
        loader.chunk_overlap = chunk_overlap

        if _components.get("building"):
            return BuildIndexResponse(status="already_running", message="Index build already in progress")

        # clear any previous errors
        _components["last_build_error"] = None

        def _build():
            try:
                _components["building"] = True
                docs = loader.load_from_dir(pdf_dir)
                vs.build(docs)
            except Exception as ex:
                # store error message to components for inspection
                _components["last_build_error"] = str(ex)
            finally:
                _components["building"] = False

        thread = threading.Thread(target=_build, daemon=True)
        thread.start()
        return BuildIndexResponse(status="started", message="Index build started in background")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, llm_model: str = model_id, reload_model: bool = False):
    try:
        # Initialize or reuse components (avoid re-loading heavy models every request)
        if reload_model:
            comps = ensure_components(llm_model=llm_model, reload=True)
        else:
            comps = _components if _components.get("llm") is not None else ensure_components(llm_model=llm_model)

        agent_sql: SQLAgent = comps["agent_sql"]
        rag: RAG = comps.get("rag")

        # Classify the question (heuristic). If it looks like a DB question and the SQL agent is ready,
        # use the SQL agent for authoritative answers. Otherwise use RAG (vector retrieval + LLM).
        use_sql = is_structured_query(req.query) and agent_sql and agent_sql.is_ready()

        if use_sql:
            print("Routing to SQL agent")
            ans = agent_sql.answer(req.query)
        else:
            print("Routing to RAG/vector retrieval")
            if rag:
                ans = rag.answer(req.query, k=req.k)
            else:
                # fallback: use SQL agent if RAG not available
                ans = agent_sql.answer(req.query)

        return QueryResponse(answer=ans)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
async def reload_model(req: ReloadModelRequest):
    """Reload the LLM with a new model name and/or max_new_tokens.
    Call with JSON { "model_name": "Qwen/...", "max_new_tokens": 200 } to apply.
    """
    try:
        model_name = req.model_name or model_id
        max_tokens = req.max_new_tokens if req.max_new_tokens is not None else 60
        ensure_components(llm_model=model_name, reload=True, max_new_tokens=max_tokens)
        return {"status": "reloaded", "model_name": model_name, "max_new_tokens": max_tokens}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index-status")
async def index_status(vectorstore: str = "vectorstores"):
    """Return quick status about index building and availability"""
    status = {
        "building": bool(_components.get("building", False)),
        "has_index": False,
        "last_build_error": _components.get("last_build_error"),
    }
    vs = _components.get("vs")
    if vs and vs.db is not None:
        status["has_index"] = True
    else:
        # check on disk
        status["has_index"] = False
        try:
            idx_dir = vectorstore
            if os.path.isdir(idx_dir) and len(os.listdir(idx_dir)) > 0:
                status["has_index"] = True
        except Exception:
            pass

    return status

