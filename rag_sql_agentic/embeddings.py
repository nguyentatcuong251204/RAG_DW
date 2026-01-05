from typing import Iterable, List
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    """Wraps a HuggingFace embedding model for use in the RAG pipeline."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self._model = HuggingFaceEmbeddings(model_name=self.model_name, cache_folder="D:/hf_cache")

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._model.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)

    @property
    def raw(self):
        return self._model
