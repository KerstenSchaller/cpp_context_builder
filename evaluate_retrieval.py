"""
Evaluate retrieval quality across one or more Chroma collections.

Expected query file format (JSON):
{
  "queries": [
    {
      "id": "q1",
      "text": "where is aquifer draining handled",
      "relevant_symbols": [
        "aquifer_drain(DFHack::color_ostream &, std::string, int, int, int, int, bool)"
      ]
    },
    {
      "id": "q2",
      "text": "tile wetness check",
      "relevance": {
        "is_wet(int16_t, int16_t, int16_t)": 2,
        "is_aquifer(int16_t, int16_t, int16_t, df::tile_designation *)": 1
      }
    }
  ]
}

Usage example:
  python evaluate_retrieval.py \
    --db-dir ./chroma_db \
    --collection cpp_callgraph \
    --collection cpp_callgraph_hybrid \
    --queries ./eval_queries.json \
    --k-values 1,3,5,10
"""

import argparse
import json
import math
import os
import statistics
import time
from typing import Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_k_values(raw: str) -> List[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        val = int(chunk)
        if val <= 0:
            raise ValueError("k-values must be positive integers")
        values.append(val)
    if not values:
        raise ValueError("No valid k-values provided")
    return sorted(set(values))


def load_queries(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data.get("queries", [])
    if not queries:
        raise ValueError("Query file does not contain any entries in 'queries'")

    normalized = []
    for i, q in enumerate(queries, start=1):
        qid = q.get("id", f"q{i}")
        text = q.get("text", "").strip()
        if not text:
            continue

        relevance_map = {}
        if isinstance(q.get("relevance"), dict):
            for symbol, grade in q["relevance"].items():
                try:
                    g = float(grade)
                except Exception:
                    continue
                if g > 0:
                    relevance_map[symbol] = g
        else:
            for symbol in q.get("relevant_symbols", []):
                relevance_map[symbol] = 1.0

        if not relevance_map:
            continue

        normalized.append({"id": qid, "text": text, "relevance": relevance_map})

    if not normalized:
        raise ValueError("No valid queries with relevance labels found")

    return normalized


def discount(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def dcg_at_k(ranked_symbols: List[str], relevance: Dict[str, float], k: int) -> float:
    score = 0.0
    for idx, symbol in enumerate(ranked_symbols[:k], start=1):
        rel = relevance.get(symbol, 0.0)
        if rel > 0:
            score += rel * discount(idx)
    return score


def idcg_at_k(relevance: Dict[str, float], k: int) -> float:
    ideal = sorted(relevance.values(), reverse=True)
    score = 0.0
    for idx, rel in enumerate(ideal[:k], start=1):
        score += rel * discount(idx)
    return score


def run_query(
    collection,
    embedding: List[float],
    n_results: int,
    merge_by_symbol: bool,
) -> Tuple[List[dict], float]:
    t0 = time.perf_counter()
    res = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["metadatas", "distances"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    rows = []
    for meta, dist in zip(metas, dists):
        if not meta:
            continue
        rows.append({
            "symbol": meta.get("symbol", ""),
            "distance": float(dist),
            "view": meta.get("view", "legacy"),
            "meta": meta,
        })

    if not merge_by_symbol:
        return rows, elapsed_ms

    merged = {}
    for row in rows:
        symbol = row["symbol"]
        if not symbol:
            continue
        prev = merged.get(symbol)
        if prev is None or row["distance"] < prev["distance"]:
            merged[symbol] = {
                "symbol": symbol,
                "distance": row["distance"],
                "views": {row["view"]},
                "meta": row["meta"],
            }
        else:
            prev["views"].add(row["view"])

    merged_rows = list(merged.values())
    merged_rows.sort(key=lambda x: x["distance"])
    return merged_rows, elapsed_ms


def collection_coverage_stats(collection) -> dict:
    total = int(collection.count())
    if total == 0:
        return {
            "total_vectors": 0,
            "source_vectors": 0,
            "source_with_code": 0,
            "source_coverage": 0.0,
        }

    limit = min(total, 100000)
    sample = collection.peek(limit=limit)
    metas = sample.get("metadatas", []) or []

    source_vectors = 0
    source_with_code = 0
    for meta in metas:
        if not meta:
            continue
        if meta.get("view") == "source":
            source_vectors += 1
            if int(meta.get("has_source", 0)) == 1:
                source_with_code += 1

    coverage = (source_with_code / source_vectors) if source_vectors > 0 else 0.0
    return {
        "total_vectors": total,
        "source_vectors": source_vectors,
        "source_with_code": source_with_code,
        "source_coverage": round(coverage, 4),
    }


def evaluate_collection(collection, q_embs, queries, k_values, merge_by_symbol):
    max_k = max(k_values)
    internal_k = max_k * 4 if merge_by_symbol else max_k

    latency_ms = []
    per_query_debug = []

    metrics = {
        f"recall@{k}": [] for k in k_values
    }
    metrics.update({f"mrr@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    for q, emb in zip(queries, q_embs):
        rows, elapsed = run_query(collection, emb, internal_k, merge_by_symbol)
        latency_ms.append(elapsed)

        ranked_symbols = [r["symbol"] for r in rows if r.get("symbol")]
        relevance = q["relevance"]

        first_rel_rank = 0
        for idx, symbol in enumerate(ranked_symbols, start=1):
            if symbol in relevance and relevance[symbol] > 0:
                first_rel_rank = idx
                break

        for k in k_values:
            topk = ranked_symbols[:k]
            hit = any(sym in relevance for sym in topk)
            metrics[f"recall@{k}"].append(1.0 if hit else 0.0)

            if first_rel_rank > 0 and first_rel_rank <= k:
                metrics[f"mrr@{k}"].append(1.0 / first_rel_rank)
            else:
                metrics[f"mrr@{k}"].append(0.0)

            denom = idcg_at_k(relevance, k)
            if denom == 0:
                metrics[f"ndcg@{k}"].append(0.0)
            else:
                metrics[f"ndcg@{k}"].append(dcg_at_k(topk, relevance, k) / denom)

        per_query_debug.append({
            "query_id": q["id"],
            "first_relevant_rank": first_rel_rank,
            "top_symbols": ranked_symbols[:max_k],
        })

    summary = {}
    for key, values in metrics.items():
        summary[key] = round(sum(values) / len(values), 4)

    summary["latency_ms_avg"] = round(sum(latency_ms) / len(latency_ms), 2)
    summary["latency_ms_p95"] = round(statistics.quantiles(latency_ms, n=20)[18], 2) if len(latency_ms) >= 20 else round(max(latency_ms), 2)

    return summary, per_query_debug


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval KPI across collections")
    parser.add_argument("--db-dir", required=True, help="Path to Chroma DB directory")
    parser.add_argument(
        "--collection",
        action="append",
        required=True,
        help="Collection name to evaluate (repeat for multiple)",
    )
    parser.add_argument("--queries", required=True, help="Path to eval queries JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--k-values", default="1,3,5,10", help="Comma separated list, e.g. 1,3,5,10")
    parser.add_argument(
        "--merge-by-symbol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge duplicate symbol hits from graph/source docs",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path")

    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    queries = load_queries(args.queries)

    model = SentenceTransformer(args.model)
    q_embs = model.encode([q["text"] for q in queries], normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=args.db_dir)

    all_results = {
        "queries": [{"id": q["id"], "text": q["text"]} for q in queries],
        "k_values": k_values,
        "collections": {},
    }

    print(f"Loaded {len(queries)} labeled queries")
    print(f"k-values: {k_values}")

    for name in args.collection:
        collection = client.get_collection(name=name)
        coverage = collection_coverage_stats(collection)
        summary, debug_rows = evaluate_collection(
            collection=collection,
            q_embs=q_embs,
            queries=queries,
            k_values=k_values,
            merge_by_symbol=args.merge_by_symbol,
        )

        all_results["collections"][name] = {
            "coverage": coverage,
            "metrics": summary,
            "per_query": debug_rows,
        }

        print(f"\nCollection: {name}")
        print(f"  vectors: {coverage['total_vectors']}")
        print(f"  source coverage: {coverage['source_coverage']}")
        for k in k_values:
            print(f"  recall@{k}: {summary[f'recall@{k}']}")
            print(f"  mrr@{k}: {summary[f'mrr@{k}']}")
            print(f"  ndcg@{k}: {summary[f'ndcg@{k}']}")
        print(f"  latency_ms_avg: {summary['latency_ms_avg']}")
        print(f"  latency_ms_p95: {summary['latency_ms_p95']}")

    if args.out:
        out_path = os.path.abspath(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()
