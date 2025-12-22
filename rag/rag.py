from typing import List
from .loader import DocumentLoader
from .embeddings import EmbeddingModel
from .vectorstore import VectorStore
from .llm import LLMModel
from langchain_core.prompts import PromptTemplate


class RAG:
    """High-level RAG orchestrator that builds the index and answers questions."""

    def __init__(
        self,
        loader: DocumentLoader,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        llm: LLMModel,
        snippet_limit: int = 512,
    ):
        self.loader = loader
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.snippet_limit = snippet_limit

    def build_index(self, pdf_dir: str):
        docs = self.loader.load_from_dir(pdf_dir)
        self.vector_store.build(docs)

    def load_index(self):
        self.vector_store.load()

    def _build_context(self, docs: List[object]) -> str:
        snippets = []
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", f"doc_{i+1}")
            text = " ".join(doc.page_content.split())
            if len(text) > self.snippet_limit:
                text = text[: self.snippet_limit] + "..."
            snippets.append(f"[{i+1}] Source: {src}\n{text}")
        return "\n\n---\n\n".join(snippets)

    def answer(self, question: str, k: int = 5) -> str:
        results = self.vector_store.similarity_search(question, k=k)
        context = self._build_context(results)

        prompt = f"""You are a helpful assistant that answers questions based on provided context.\n{context}\nAnswer the question: {question}"""

        prompt_template = PromptTemplate.from_template(
            "INSTRUCTION: {prompt} \n\n RESPONSE:"
        )
        formatted = prompt_template.format(prompt=prompt)
        response = self.llm.generate(prompt=formatted)
        return response
