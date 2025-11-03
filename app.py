import json
import os
import argparse

from src.rag.config import RagConfig
from src.rag.pipeline import RagPipeline


def cmd_build(args):
    cfg = RagConfig(index_dir=args.index_dir)
    pipe = RagPipeline(cfg)
    pipe.build_index(args.docs_dir)
    print(f"Index built and saved to: {args.index_dir}")


def cmd_query(args):
    cfg = RagConfig(index_dir=args.index_dir)
    pipe = RagPipeline(cfg)
    pipe.load_index()
    out = pipe.answer(args.question, top_k=args.top_k)
    print(json.dumps({
        "question": out["question"],
        "answer": out["answer"],
        "matches": [{
            "doc_id": m["doc_id"],
            "chunk_id": m["chunk_id"],
            "score": round(m["score"], 3),
            "source_path": m.get("source_path")
        } for m in out["passages"]]
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build", help="Build FAISS index from documents")
    p_build.add_argument("--docs_dir", required=True, help="Directory with .txt/.pdf files")
    p_build.add_argument("--index_dir", default="indexes", help="Directory to store index files")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Query the index to answer a question")
    p_query.add_argument("--index_dir", default="indexes", help="Directory with index files")
    p_query.add_argument("--question", required=True, help="User question")
    p_query.add_argument("--top_k", type=int, default=5, help="Number of passages to retrieve")
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()



