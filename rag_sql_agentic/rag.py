from typing import List
from .loader import DocumentLoader
from .embeddings import EmbeddingModel
from .vectorstore import VectorStore
from .llm import LLMModel
from langchain_core.prompts import PromptTemplate
import re


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
        # Ensure vectorstore is loaded (attempt to load from disk if necessary)
        try:
            if self.vector_store.db is None:
                try:
                    self.vector_store.load()
                except Exception:
                    # If load fails or there is no index yet, return a friendly message
                    return "No vector index found. Please run the /build-index endpoint to create the index and try again."
        except Exception:
            return "No vector index found. Please run the /build-index endpoint to create the index and try again."

        results = self.vector_store.similarity_search(question, k=k)
        context = self._build_context(results)

        prompt_template = PromptTemplate.from_template(
            "INSTRUCTION: {question}\n\nRESPONSE: Provide a concise, complete, and self-contained answer. Do not end mid-sentence. If you must abbreviate, say 'see more' and avoid trailing ellipses.\n\nRESPONSE:")
        # Use the 'question' key to match the template placeholder
        formatted = prompt_template.format(question=question, context=context)
        response = self.llm.generate(prompt=formatted)

        # Normalize response to string
        raw = response if isinstance(response, str) else str(response)

        # Try to extract everything after 'RESPONSE:' or familiar markers
        m = re.search(r"(?:RESPONSE:|Final Answer:|Answer:)\s*(.*)$", raw, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            # Strip code fences and optional 'sql' markers
            ans = re.sub(r"```\s*(?:sql)?\s*", "", ans, flags=re.IGNORECASE)
            ans = re.sub(r"```", "", ans)
            ans = re.sub(r"^\s*sql\s+", "", ans, flags=re.IGNORECASE)
            return ans.strip()

        # Fallback: return full text trimmed
        return raw.strip()
