"""Simple CLI for building an index and querying a RAG system."""
import sys
try:
    import pwd
except ImportError:
    import unittest.mock
    sys.modules["pwd"] = unittest.mock.MagicMock()

import argparse
import os
import glob
from rag import DocumentLoader, EmbeddingModel, VectorStore, LLMModel, RAG
from rag.sql_db import convert_csv_to_sqlite
from rag.agent import SQLAgent


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


def build_sql_db(args):
    """
    Converts CSV files in the data directory to a SQLite database.
    """
    data_dir = args.data_dir
    db_path = args.db_path
    
    # Find CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    # For now, convert the first one or loop. The user has one main file.
    # We'll just convert the first one found for simplicity or merging implementation.
    csv_file = csv_files[0] 
    print(f"Found CSV: {csv_file}")
    
    convert_csv_to_sqlite(csv_file, db_path, table_name="hospital")
    print(f"Database built at {db_path}")
    

def query(args):
    if args.agent_type == "sql":


        print("Initializing SQL Agent...")
        llm = LLMModel(model_name=args.llm_model, temperature=0.01, max_new_tokens=125) # Low temp for code generation
        agent = SQLAgent(db_path=args.db_path, llm_model=llm)
        
        print(f"--- ASKING SQL AGENT: {args.query} ---")
        answer = agent.answer(args.query)
        print("--- ANSWER ---")
        print(answer)
        
    else:
        # Standard RAG
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

    # Build Index (Vector)
    p_build = sub.add_parser("build-index")
    p_build.add_argument("pdf_dir", help="Directory with PDFs (default=data)", nargs="?", default="data")
    p_build.add_argument("--vectorstore", default="vectorstores")
    p_build.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p_build.add_argument("--chunk-size", type=int, default=512)
    p_build.add_argument("--chunk-overlap", type=int, default=50)
    p_build.set_defaults(func=build_index)
    
    # Build SQL DB
    p_build_sql = sub.add_parser("build-sql-db")
    p_build_sql.add_argument("data_dir", help="Directory with CSVs (default=data)", nargs="?", default="data")
    p_build_sql.add_argument("--db-path", default="hospital.db")
    p_build_sql.set_defaults(func=build_sql_db)

    # Query
    p_q = sub.add_parser("query")
    p_q.add_argument("query", help="User question to ask")
    p_q.add_argument("--agent-type", choices=["rag", "sql"], default="sql", help="Type of agent to use: rag (vector) or sql (database)")
    p_q.add_argument("--k", type=int, default=5)
    p_q.add_argument("--vectorstore", default="vectorstores")
    p_q.add_argument("--db-path", default="rag.db")
    p_q.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p_q.add_argument("--llm-model", default="microsoft/phi-2")
    p_q.set_defaults(func=query)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

