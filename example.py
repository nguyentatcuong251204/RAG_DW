from rag import DocumentLoader, EmbeddingModel, VectorStore, LLMModel, RAG


def main():
    loader = DocumentLoader()
    emb = EmbeddingModel()
    vs = VectorStore(emb)

    # Build index (reads all PDFs from data/)
    print("Building index from data/")
    docs = loader.load_from_dir("data")
    vs.build(docs)
    print("Saved vectorstore to disk")

    # Load and query
    vs.load()

    llm = LLMModel()
    rag = RAG(loader, emb, vs, llm)
    q = "How many dataset are mentioned in the document?"
    print("Query:", q)
    ans = rag.answer(q)
    print(ans)


if __name__ == "__main__":
    main()
