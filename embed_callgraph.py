"""
embed_callgraph.py

Create embeddings from callgraph.json and store them in a persistent Chroma DB.

Input format (from build_callgraph.py):
{
    "nodes": [{"id": "...", "file": "...", "line": 12}],
    "edges": [{"caller": "...", "callee": "..."}]
}

Index modes:
    - graph:  graph summary text per function
    - source: extracted function source text per function
    - hybrid: both graph + source documents (best for retrieval quality)

Usage:
    python embed_callgraph.py --callgraph callgraph.json --index-mode hybrid

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
MAX_SOURCE_LINES = 1200


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


def resolve_source_path(file_hint, source_root):
    if not file_hint:
        return None

    candidates = [file_hint]
    if source_root:
        candidates.append(os.path.join(source_root, file_hint))

    for candidate in candidates:
        abs_path = os.path.abspath(candidate)
        if os.path.exists(abs_path):
            return abs_path
    return None


def extract_function_source_text(file_path, start_line, max_lines=MAX_SOURCE_LINES):
    if not file_path or not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return None

    start_idx = max(0, int(start_line) - 1)
    start_idx = min(start_idx, len(lines) - 1)

    opened = False
    depth = 0
    in_single = False
    in_double = False
    escaped = False
    end_idx = min(len(lines), start_idx + 80)

    max_idx = min(len(lines), start_idx + max_lines)
    for i in range(start_idx, max_idx):
        line = lines[i]
        for ch in line:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue

            if in_single:
                if ch == "'":
                    in_single = False
                continue

            if in_double:
                if ch == '"':
                    in_double = False
                continue

            if ch == "'":
                in_single = True
                continue
            if ch == '"':
                in_double = True
                continue

            if ch == "{":
                opened = True
                depth += 1
            elif ch == "}" and opened:
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    return "".join(lines[start_idx:end_idx]).strip()

    return "".join(lines[start_idx:end_idx]).strip()


def make_source_doc(node, source_root):
    node_id = node.get("id", "")
    file_hint = str(node.get("file", ""))
    line = int(node.get("line", 0))

    src_path = resolve_source_path(file_hint, source_root)
    snippet = extract_function_source_text(src_path, line) if src_path else None

    if snippet:
        text = (
            f"Function: {node_id}\n"
            f"Location: {file_hint}:{line}\n"
            f"Source:\n{snippet}"
        )
        return text, True

    text = (
        f"Function: {node_id}\n"
        f"Location: {file_hint}:{line}\n"
        "Source: <unavailable>"
    )
    return text, False


def build_index_records(nodes, edges, index_mode, source_root):
    calls_map, called_by_map = build_adjacency(edges)

    ids = []
    docs = []
    metadatas = []

    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue

        file_path = str(node.get("file", ""))
        line = int(node.get("line", 0))

        graph_doc, callees, callers = make_doc(node, calls_map, called_by_map)

        if index_mode in ("graph", "hybrid"):
            graph_id = node_id if index_mode == "graph" else f"{node_id}::graph"
            ids.append(graph_id)
            docs.append(graph_doc)
            metadatas.append(
                {
                    "symbol": node_id,
                    "file": file_path,
                    "line": line,
                    "view": "graph",
                    "num_calls": len(callees),
                    "num_called_by": len(callers),
                }
            )

        if index_mode in ("source", "hybrid"):
            source_doc, has_source = make_source_doc(node, source_root)
            source_id = node_id if index_mode == "source" else f"{node_id}::source"
            ids.append(source_id)
            docs.append(source_doc)
            metadatas.append(
                {
                    "symbol": node_id,
                    "file": file_path,
                    "line": line,
                    "view": "source",
                    "num_calls": len(callees),
                    "num_called_by": len(callers),
                    "has_source": 1 if has_source else 0,
                }
            )

    return ids, docs, metadatas


def index_callgraph(
    callgraph_path,
    db_dir,
    collection_name,
    model_name,
    batch_size=128,
    index_mode="graph",
    source_root=None,
):
    nodes, edges = load_callgraph(callgraph_path)
    if source_root is None:
        source_root = os.path.dirname(os.path.abspath(callgraph_path))

    model = SentenceTransformer(model_name)
    _, collection = connect_collection(db_dir, collection_name)

    ids, docs, metadatas = build_index_records(nodes, edges, index_mode, source_root)

    if not ids:
        print("No valid nodes found in callgraph; nothing indexed.")
        return collection, 0

    print(f"Embedding {len(ids)} document(s) in {index_mode} mode with model: {model_name}")

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


def run_query(collection, model_name, query_text, top_k, merge_by_symbol=True):
    model = SentenceTransformer(model_name)
    qvec = model.encode([query_text], normalize_embeddings=True).tolist()[0]

    internal_k = top_k * 4 if merge_by_symbol else top_k

    res = collection.query(
        query_embeddings=[qvec],
        n_results=internal_k,
        include=["documents", "metadatas", "distances"],
    )

    print("\nTop results:")
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    rows = list(zip(docs, metas, dists))
    if merge_by_symbol:
        merged = {}
        for doc, meta, dist in rows:
            symbol = meta.get("symbol", "<unknown>") if meta else "<unknown>"
            view = meta.get("view", "legacy") if meta else "legacy"
            existing = merged.get(symbol)
            if existing is None or dist < existing["dist"]:
                merged[symbol] = {
                    "doc": doc,
                    "meta": meta,
                    "dist": dist,
                    "views": {view},
                }
            else:
                existing["views"].add(view)

        rows = [
            (v["doc"], v["meta"], v["dist"], sorted(v["views"]))
            for v in merged.values()
        ]
        rows.sort(key=lambda x: x[2])
        rows = rows[:top_k]
    else:
        rows = [(doc, meta, dist, [meta.get("view", "legacy") if meta else "legacy"]) for doc, meta, dist in rows[:top_k]]

    for idx, (doc, meta, dist, views) in enumerate(rows, start=1):
        symbol = meta.get("symbol", "<unknown>") if meta else "<unknown>"
        file_path = meta.get("file", "") if meta else ""
        line = meta.get("line", 0) if meta else 0
        print(f"\n[{idx}] {symbol}")
        print(f"  distance: {dist:.4f}")
        print(f"  location: {file_path}:{line}")
        print(f"  views: {', '.join(views)}")
        preview = doc.splitlines()[:6]
        print("  preview:")
        for line_text in preview:
            print(f"    {line_text}")


def validate_db(
    db_dir,
    collection_name,
    model_name=None,
    query_text=None,
    top_k=5,
    merge_by_symbol=True,
):
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
        run_query(collection, model_name, query_text, top_k, merge_by_symbol=merge_by_symbol)

    return count


def main():
    parser = argparse.ArgumentParser(description="Embed callgraph nodes into Chroma")
    parser.add_argument("--callgraph", default="callgraph.json", help="Path to callgraph JSON")
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR, help="Chroma persistent directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    parser.add_argument(
        "--index-mode",
        choices=["graph", "source", "hybrid"],
        default="graph",
        help="What to index: graph summaries, source text, or both",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Project root for resolving node file paths to source files",
    )

    parser.add_argument("--query", default=None, help="Optional semantic query after indexing")
    parser.add_argument("--top-k", type=int, default=5, help="Top K query results")
    parser.add_argument(
        "--merge-by-symbol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge graph/source hits by function symbol at query time",
    )
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
            merge_by_symbol=args.merge_by_symbol,
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
        index_mode=args.index_mode,
        source_root=args.source_root,
    )

    print(f"\nDone. Indexed {count} node(s) into collection '{args.collection}'.")
    print(f"Chroma DB path: {os.path.abspath(args.db_dir)}")

    if args.query:
        run_query(
            collection,
            args.model,
            args.query,
            args.top_k,
            merge_by_symbol=args.merge_by_symbol,
        )


if __name__ == "__main__":
    main()
