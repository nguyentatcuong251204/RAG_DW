"""Simple CLI for building an index and querying a RAG system."""
import argparse
from rag import DocumentLoader, EmbeddingModel, VectorStore, LLMModel, RAG


def build_index(args):
    loader = DocumentLoader(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    emb = EmbeddingModel(model_name=args.embedding_model)
    vs = VectorStore(emb, store_dir=args.vectorstore)

    # llm = LLMModel(model_name=args.llm_model)
    llm = None
    rag = RAG(loader, emb, vs, llm)

    print(f"Building index from {args.pdf_dir} -> {args.vectorstore}")
    rag.build_index(args.pdf_dir)
    print("Done building index.")


def query(args):
    loader = DocumentLoader()
    emb = EmbeddingModel(model_name=args.embedding_model)
    vs = VectorStore(emb, store_dir=args.vectorstore)
    vs.load()

    llm = LLMModel(model_name=args.llm_model)
    rag = RAG(loader, emb, vs, llm)

    answer = rag.answer(args.query, k=args.k)
    print("--- ANSWER ---")
    print(answer)


def main():
    parser = argparse.ArgumentParser(description="Simple RAG CLI")
    sub = parser.add_subparsers()

    p_build = sub.add_parser("build-index")
    p_build.add_argument("pdf_dir", help="Directory with PDFs (default=data)", nargs="?", default="data")
    p_build.add_argument("--vectorstore", default="vectorstores")
    p_build.add_argument("--llm-model", default="microsoft/phi-2")
    p_build.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p_build.add_argument("--chunk-size", type=int, default=512)
    p_build.add_argument("--chunk-overlap", type=int, default=50)
    p_build.set_defaults(func=build_index)

    p_q = sub.add_parser("query")
    p_q.add_argument("query", help="User question to ask")
    p_q.add_argument("--k", type=int, default=5)
    p_q.add_argument("--vectorstore", default="vectorstores")
    p_q.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p_q.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    p_q.set_defaults(func=query)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
