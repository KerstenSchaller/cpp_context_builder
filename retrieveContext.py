"""
Retrieve C++ callgraph context from Chroma DB for GitHub Copilot usage.

Usage:
  python3 retrieveContext.py --query "aquifer draining" --top-k 5
"""

import argparse
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_DB_DIR = "./chroma_db"
DEFAULT_COLLECTION = "cpp_callgraph"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def retrieve_callgraph(
    db_dir: str,
    collection_name: str,
    query: str,
    embed_model: str,
    top_k: int,
    merge_by_symbol: bool,
) -> List[Dict[str, Any]]:
    # Embed the query
    model = SentenceTransformer(embed_model)
    qvec = model.encode([query], normalize_embeddings=True).tolist()[0]

    # Access Chroma DB
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(name=collection_name)

    # Optionally fetch more results for merging
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
                "snippet": doc or "",
            }
        )

    if merge_by_symbol:
        merged = {}
        for r in rows:
            key = r["symbol"]
            prev = merged.get(key)
            if prev is None or r["distance"] < prev["distance"]:
                merged[key] = {**r, "views": {r["view"]}}
            else:
                prev["views"].add(r["view"])
        merged_rows = list(merged.values())
        for r in merged_rows:
            r["views"] = sorted(list(r.get("views", [])))
        merged_rows.sort(key=lambda x: x["distance"])
        return merged_rows[:top_k]

    return rows[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Retrieve C++ callgraph for Copilot")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--merge-by-symbol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    results = retrieve_callgraph(
        db_dir=args.db_dir,
        collection_name=args.collection,
        query=args.query,
        embed_model=args.embed_model,
        top_k=args.top_k,
        merge_by_symbol=args.merge_by_symbol,
    )

    # Output structured JSON-like data
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
