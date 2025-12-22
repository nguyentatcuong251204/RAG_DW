from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document


class DocumentLoader:
    """Loads PDF documents from a directory and splits them into chunks."""

    def __init__(self, chunk_size: int = 215, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_from_dir(self, pdf_dir: str, glob: str = "*.pdf") -> List[Document]:
        loader = DirectoryLoader(pdf_dir, glob=glob, loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def load_single_pdf(self, filepath: str) -> List[Document]:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(docs)
