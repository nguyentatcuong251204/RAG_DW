from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, DirectoryLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document




class DocumentLoader:
    """Loads CSV tables from a directory (no PDF)."""

    def __init__(self):
        pass

    def load_from_dir(self, directory: str) -> List[Document]:
        documents = []

        try:
            csv_loader = DirectoryLoader(
                directory,
                glob="*.csv",
                loader_cls=CSVLoader
            )
            csv_docs = csv_loader.load()

            for doc in csv_docs:
                # Mark this document as table data
                doc.metadata["type"] = "table"
                documents.append(doc)

        except Exception as e:
            print(f"CSV load error: {e}")

        return documents

