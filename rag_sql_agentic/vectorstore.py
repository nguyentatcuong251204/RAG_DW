from typing import List, Optional, Any
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document



import os

class VectorStore:
    """Manages FAISS vector store creation, saving, loading and retrieval."""

    def __init__(self, embedding_model, store_dir: str = "vectorstores"):
        self.embedding_model = embedding_model
        self.store_dir = store_dir
        self._db: Optional[FAISS] = None

    def build(self, docs: List[Document]):
        # ensure directory exists to avoid silent failures when saving
        os.makedirs(self.store_dir, exist_ok=True)
        self._db = FAISS.from_documents(docs, self.embedding_model.raw)
        self._db.save_local(self.store_dir)
        return self._db

    def load(self, allow_dangerous_deserialization: bool = True):
        self._db = FAISS.load_local(
            self.store_dir,
            self.embedding_model.raw,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        )
        return self._db

    def similarity_search(self, query: str, k: int = 4):
        if self._db is None:
            raise ValueError("Vector store not loaded. Call load() or build() first.")
        return self._db.similarity_search(query, k=k)

    @property
    def db(self):
        return self._db
