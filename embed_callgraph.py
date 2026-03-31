"""
embed_callgraph.py

Create embeddings from callgraph.json and store them in a persistent Chroma DB.

Input format (from build_callgraph.py):
{
  "nodes": [{"id": "...", "file": "...", "line": 12}],
  "edges": [{"caller": "...", "callee": "..."}]
}

Usage:
  python embed_callgraph.py --callgraph callgraph.json

Optional query:
  python embed_callgraph.py --callgraph callgraph.json --query "magma flow" --top-k 5
"""

import argparse
import json
import os
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "cpp_callgraph"
DEFAULT_DB_DIR = "./chroma_db"


def connect_collection(db_dir, collection_name):
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(name=collection_name)
    return client, collection


def load_callgraph(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    return nodes, edges


def build_adjacency(edges):
    calls_map = defaultdict(list)
    called_by_map = defaultdict(list)

    for e in edges:
        caller = e.get("caller")
        callee = e.get("callee")
        if not caller or not callee:
            continue
        calls_map[caller].append(callee)
        called_by_map[callee].append(caller)

    return calls_map, called_by_map


def make_doc(node, calls_map, called_by_map):
    node_id = node.get("id", "")
    file_path = node.get("file", "")
    line = node.get("line", 0)

    callees = sorted(set(calls_map.get(node_id, [])))
    callers = sorted(set(called_by_map.get(node_id, [])))

    # A retrieval-friendly text view that preserves both local and graph context.
    lines = [
        f"Function: {node_id}",
        f"Location: {file_path}:{line}",
        f"Calls count: {len(callees)}",
        f"Called-by count: {len(callers)}",
        "Calls:",
    ]

    if callees:
        lines.extend(f"- {c}" for c in callees)
    else:
        lines.append("- <none>")

    lines.append("Called by:")
    if callers:
        lines.extend(f"- {c}" for c in callers)
    else:
        lines.append("- <none>")

    return "\n".join(lines), callees, callers


def index_callgraph(callgraph_path, db_dir, collection_name, model_name, batch_size=128):
    nodes, edges = load_callgraph(callgraph_path)
    calls_map, called_by_map = build_adjacency(edges)

    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(name=collection_name)

    ids = []
    docs = []
    metadatas = []

    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue

        doc, callees, callers = make_doc(node, calls_map, called_by_map)
        ids.append(node_id)
        docs.append(doc)
        metadatas.append(
            {
                "symbol": node_id,
                "file": str(node.get("file", "")),
                "line": int(node.get("line", 0)),
                "num_calls": len(callees),
                "num_called_by": len(callers),
            }
        )

    if not ids:
        print("No valid nodes found in callgraph; nothing indexed.")
        return collection, 0

    print(f"Embedding {len(ids)} function node(s) with model: {model_name}")

    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = docs[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]

        embeddings = model.encode(batch_docs, normalize_embeddings=True).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
        )

        print(f"  Indexed {min(i + batch_size, len(ids))}/{len(ids)}")

    return collection, len(ids)


def run_query(collection, model_name, query_text, top_k):
    model = SentenceTransformer(model_name)
    qvec = model.encode([query_text], normalize_embeddings=True).tolist()[0]

    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    print("\nTop results:")
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        symbol = meta.get("symbol", "<unknown>") if meta else "<unknown>"
        file_path = meta.get("file", "") if meta else ""
        line = meta.get("line", 0) if meta else 0
        print(f"\n[{idx}] {symbol}")
        print(f"  distance: {dist:.4f}")
        print(f"  location: {file_path}:{line}")
        preview = doc.splitlines()[:6]
        print("  preview:")
        for line_text in preview:
            print(f"    {line_text}")


def validate_db(db_dir, collection_name, model_name=None, query_text=None, top_k=5):
    client, collection = connect_collection(db_dir, collection_name)

    collections = [c.name for c in client.list_collections()]
    print("DB health report:")
    print(f"  db_dir: {os.path.abspath(db_dir)}")
    print(f"  collections: {collections}")
    print(f"  active collection: {collection_name}")

    count = collection.count()
    print(f"  vector count: {count}")

    sample = collection.peek(limit=min(5, max(1, count))) if count > 0 else {"ids": []}
    print(f"  sample ids: {sample.get('ids', [])}")

    if query_text:
        if not model_name:
            raise ValueError("model_name is required when running a semantic query")
        print("\nRunning semantic query as part of validation...")
        run_query(collection, model_name, query_text, top_k)

    return count


def main():
    parser = argparse.ArgumentParser(description="Embed callgraph nodes into Chroma")
    parser.add_argument("--callgraph", default="callgraph.json", help="Path to callgraph JSON")
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR, help="Chroma persistent directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")

    parser.add_argument("--query", default=None, help="Optional semantic query after indexing")
    parser.add_argument("--top-k", type=int, default=5, help="Top K query results")
    parser.add_argument(
        "--validate-db",
        action="store_true",
        help="Validate existing DB and exit (no indexing)",
    )

    args = parser.parse_args()

    if args.validate_db:
        validate_db(
            db_dir=args.db_dir,
            collection_name=args.collection,
            model_name=args.model,
            query_text=args.query,
            top_k=args.top_k,
        )
        return

    callgraph_abs = os.path.abspath(args.callgraph)
    if not os.path.exists(callgraph_abs):
        raise FileNotFoundError(f"Callgraph file not found: {callgraph_abs}")

    collection, count = index_callgraph(
        callgraph_path=callgraph_abs,
        db_dir=args.db_dir,
        collection_name=args.collection,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    print(f"\nDone. Indexed {count} node(s) into collection '{args.collection}'.")
    print(f"Chroma DB path: {os.path.abspath(args.db_dir)}")

    if args.query:
        run_query(collection, args.model, args.query, args.top_k)


if __name__ == "__main__":
    main()
