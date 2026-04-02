"""
RAG QA over Chroma code index.

Flow:
1) Embed user question
2) Retrieve top-k docs from Chroma
3) Build grounded prompt with citations
4) Call an OpenAI-compatible chat API

Environment variables:
- LLM_API_KEY        (required unless your endpoint does not require auth)
- LLM_BASE_URL       (optional, default: https://api.openai.com/v1)

Examples:
  python rag_answer.py --question "Where is aquifer draining handled?"

  python rag_answer.py \
    --question "How is wet tile detection implemented?" \
    --collection cpp_callgraph_hybrid \
    --top-k 10 \
    --model gpt-4o-mini
"""

import argparse
import os
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_DB_DIR = "./chroma_db"
DEFAULT_COLLECTION = "cpp_callgraph_hybrid"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4o-mini"


def retrieve_context(
    db_dir: str,
    collection_name: str,
    query: str,
    embed_model: str,
    top_k: int,
    merge_by_symbol: bool,
) -> List[Dict[str, Any]]:
    model = SentenceTransformer(embed_model)
    qvec = model.encode([query], normalize_embeddings=True).tolist()[0]

    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name=collection_name)

    internal_k = top_k * 4 if merge_by_symbol else top_k
    res = collection.query(
        query_embeddings=[qvec],
        n_results=internal_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    rows = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        rows.append(
            {
                "symbol": meta.get("symbol", "UNKNOWN"),
                "file": meta.get("file", ""),
                "line": meta.get("line", 0),
                "view": meta.get("view", "legacy"),
                "distance": float(dist),
                "document": doc or "",
            }
        )

    if not merge_by_symbol:
        return rows[:top_k]

    merged = {}
    for r in rows:
        key = r["symbol"]
        prev = merged.get(key)
        if prev is None or r["distance"] < prev["distance"]:
            merged[key] = {
                **r,
                "views": {r["view"]},
            }
        else:
            prev["views"].add(r["view"])

    merged_rows = list(merged.values())
    merged_rows.sort(key=lambda x: x["distance"])
    for r in merged_rows:
        r["views"] = sorted(list(r.get("views", [])))
    return merged_rows[:top_k]


def build_prompt(question: str, ctx_rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("You are a senior C++ code assistant. Answer only from retrieved context.")
    lines.append("If context is insufficient, say exactly what is missing.")
    lines.append("Cite symbols and file:line in your answer.")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Retrieved context:")

    for i, row in enumerate(ctx_rows, start=1):
        lines.append(f"[{i}] Symbol: {row['symbol']}")
        lines.append(f"    Location: {row['file']}:{row['line']}")
        if "views" in row:
            lines.append(f"    Views: {', '.join(row['views'])}")
        else:
            lines.append(f"    View: {row['view']}")
        lines.append(f"    Distance: {row['distance']:.4f}")
        doc_preview = "\n".join((row["document"] or "").splitlines()[:40])
        lines.append("    Snippet:")
        for snippet_line in doc_preview.splitlines():
            lines.append(f"      {snippet_line}")
        lines.append("")

    lines.append("Task:")
    lines.append("1) Explain the answer clearly.")
    lines.append("2) Mention likely call path if inferable.")
    lines.append("3) End with citations like: Symbol (file:line).")
    return "\n".join(lines)


def ask_llm(prompt: str, model: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        raise RuntimeError("Set LLM_API_KEY (or OPENAI_API_KEY) before running.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Answer with grounded code reasoning and citations."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask an LLM over a Chroma code index")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR, help="Chroma DB directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k retrieved symbols")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="LLM model name")
    parser.add_argument(
        "--merge-by-symbol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge graph/source hits by symbol",
    )
    parser.add_argument(
        "--show-context-only",
        action="store_true",
        help="Print assembled prompt/context without calling LLM",
    )

    args = parser.parse_args()

    rows = retrieve_context(
        db_dir=args.db_dir,
        collection_name=args.collection,
        query=args.question,
        embed_model=args.embed_model,
        top_k=max(1, args.top_k),
        merge_by_symbol=args.merge_by_symbol,
    )

    if not rows:
        raise RuntimeError("No retrieval results. Check db-dir/collection.")

    prompt = build_prompt(args.question, rows)

    print("=== Retrieval hits ===")
    for i, r in enumerate(rows, start=1):
        print(f"{i}. {r['symbol']} | {r['file']}:{r['line']} | d={r['distance']:.4f}")

    if args.show_context_only:
        print("\n=== Prompt sent to LLM ===\n")
        print(prompt)
        return

    answer = ask_llm(prompt, model=args.model)
    print("\n=== LLM answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
