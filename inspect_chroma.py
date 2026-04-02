"""
Inspect a Chroma DB directory from the command line.

Examples:
  python inspect_chroma.py --db-dir ./chroma_db
  python inspect_chroma.py --db-dir ./chroma_db --collection cpp_callgraph --sample 5
  python inspect_chroma.py --db-dir ./chroma_db --collection cpp_callgraph --query "aquifer flow" --top-k 5
"""

import argparse
import csv
import os

import chromadb


def list_collections(client):
    cols = client.list_collections()
    if not cols:
        print("No collections found.")
        return []

    names = [c.name for c in cols]
    print("Collections:")
    for n in names:
        print(f"  - {n}")
    return names


def write_csv(path, rows):
    if not rows:
        return

    fields = [
        "collection",
        "row_type",
        "rank",
        "id",
        "symbol",
        "view",
        "file",
        "line",
        "distance",
        "preview",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_csv_path(base_path, collection_name):
    root, ext = os.path.splitext(base_path)
    ext = ext or ".csv"
    return f"{root}.{collection_name}{ext}"


def print_collection_summary(client, name, sample_size):
    collection = client.get_collection(name=name)
    count = int(collection.count())

    print(f"\nCollection: {name}")
    print(f"  vector_count: {count}")

    sample_rows = []

    if count == 0:
        print("  sample_ids: []")
        return collection, sample_rows

    sample = collection.peek(limit=min(sample_size, count))
    ids = sample.get("ids", [])
    docs = sample.get("documents", [])
    metas = sample.get("metadatas", [])

    print("  sample_rows:")
    for i, cid in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        doc = docs[i] if i < len(docs) else ""
        symbol = meta.get("symbol", "") if meta else ""
        view = meta.get("view", "legacy") if meta else "legacy"
        location = f"{meta.get('file', '')}:{meta.get('line', 0)}" if meta else ""
        preview = ""
        if doc:
            first = doc.splitlines()[0].strip()
            preview = first[:120]

        print(f"    - id: {cid}")
        print(f"      symbol: {symbol}")
        print(f"      view: {view}")
        print(f"      location: {location}")
        print(f"      preview: {preview}")

        sample_rows.append(
            {
                "collection": name,
                "row_type": "sample",
                "rank": i + 1,
                "id": cid,
                "symbol": symbol,
                "view": view,
                "file": meta.get("file", "") if meta else "",
                "line": meta.get("line", 0) if meta else 0,
                "distance": "",
                "preview": preview,
            }
        )

    return collection, sample_rows


def query_collection(collection, collection_name, query_text, top_k):
    print(f"\nSemantic query: {query_text}")
    # Let Chroma embed query text using the collection's embedding function if available.
    result = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    rows = []

    if not docs:
        print("  No results returned.")
        return rows

    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        symbol = meta.get("symbol", "") if meta else ""
        view = meta.get("view", "legacy") if meta else "legacy"
        location = f"{meta.get('file', '')}:{meta.get('line', 0)}" if meta else ""
        print(f"  [{idx}] {symbol}")
        print(f"      distance: {dist:.4f}")
        print(f"      view: {view}")
        print(f"      location: {location}")
        if doc:
            preview = "\n".join(doc.splitlines()[:4])
            print("      preview:")
            for line in preview.splitlines():
                print(f"        {line}")
        else:
            preview = ""

        rows.append(
            {
                "collection": collection_name,
                "row_type": "query",
                "rank": idx,
                "id": "",
                "symbol": symbol,
                "view": view,
                "file": meta.get("file", "") if meta else "",
                "line": meta.get("line", 0) if meta else 0,
                "distance": round(float(dist), 6),
                "preview": preview,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Inspect a Chroma persistent DB")
    parser.add_argument("--db-dir", required=True, help="Path to Chroma DB directory")
    parser.add_argument("--collection", default=None, help="Specific collection to inspect")
    parser.add_argument("--sample", type=int, default=5, help="Number of sample rows per collection")
    parser.add_argument("--query", default=None, help="Optional semantic query text")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k results for query")
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV output path. If multiple collections are inspected, files are suffixed per collection.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.db_dir):
        raise FileNotFoundError(f"DB directory not found: {os.path.abspath(args.db_dir)}")

    client = chromadb.PersistentClient(path=args.db_dir)
    names = list_collections(client)
    if not names:
        return

    target = [args.collection] if args.collection else names

    for name in target:
        if name not in names:
            print(f"\nCollection not found: {name}")
            continue

        col, sample_rows = print_collection_summary(client, name, sample_size=max(1, args.sample))
        query_rows = []

        if args.query:
            query_rows = query_collection(col, name, args.query, top_k=max(1, args.top_k))

        if args.csv_out:
            out_path = args.csv_out
            if len(target) > 1:
                out_path = resolve_csv_path(args.csv_out, name)

            write_csv(out_path, sample_rows + query_rows)
            print(f"\nCSV written: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
