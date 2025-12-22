from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag import DocumentLoader, EmbeddingModel, VectorStore, LLMModel, RAG
import threading
import os

app = FastAPI(title="RAG Chat API")

# Serve the simple frontend from /
# if os.path.isdir("web_ui"):
#     app.mount("/", StaticFiles(directory="web_ui", html=True), name="web_ui")
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
    "rag": None,
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


def ensure_components(embedding_model: str = "BAAI/bge-base-en-v1.5", llm_model: str = "Qwen/Qwen2.5-3B-Instruct", store_dir: str = "vectorstores"):
    if _components["loader"] is None:
        _components["loader"] = DocumentLoader()
    if _components["emb"] is None:
        _components["emb"] = EmbeddingModel(model_name=embedding_model)
    if _components["vs"] is None:
        _components["vs"] = VectorStore(_components["emb"], store_dir=store_dir)
    if _components["llm"] is None:
        _components["llm"] = LLMModel(model_name=llm_model)
    if _components["rag"] is None:
        _components["rag"] = RAG(_components["loader"], _components["emb"], _components["vs"], _components["llm"])
    return _components


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/build-index", response_model=BuildIndexResponse)
async def build_index(pdf_dir: str = "data", vectorstore: str = "vectorstores", chunk_size: int = 215, chunk_overlap: int = 50, embedding_model: str = "BAAI/bge-base-en-v1.5"):
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
async def query(req: QueryRequest, vectorstore: str = "vectorstores", embedding_model: str = "BAAI/bge-base-en-v1.5", llm_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    try:
        comps = ensure_components(embedding_model=embedding_model, llm_model=llm_model, store_dir=vectorstore)
        vs: VectorStore = comps["vs"]
        # load vectorstore if not loaded
        if vs.db is None:
            vs.load()
        rag: RAG = comps["rag"]
        ans = rag.answer(req.query, k=req.k)
        return QueryResponse(answer=ans)
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