"""Retrieval metrics at the answer (pair) level.

Chunks are mapped back to the MedQuAD answer they came from, duplicates are
dropped (first occurrence keeps its rank), then standard IR metrics are computed
against graded judgements: gain 3 = excellent, 2 = incomplete, 1 = related.
Recall and MRR count an answer as relevant when gain >= `rel_threshold`.
"""
from __future__ import annotations

import math
import time

import numpy as np


def dedupe_pairs(hits: list[tuple[int, float]], chunk_pair: list[str]) -> list[str]:
    seen, ranked = set(), []
    for i, _ in hits:
        p = chunk_pair[i]
        if p not in seen:
            seen.add(p)
            ranked.append(p)
    return ranked


def ndcg_at(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    dcg = sum((2 ** qrels.get(p, 0) - 1) / math.log2(r + 2) for r, p in enumerate(ranked[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at(ranked, qrels, k, thr):
    rel = {p for p, g in qrels.items() if g >= thr}
    return len(rel & set(ranked[:k])) / len(rel) if rel else 0.0


def hit_at(ranked, qrels, k, thr):
    return float(any(qrels.get(p, 0) >= thr for p in ranked[:k]))


def mrr(ranked, qrels, thr, k=10):
    for r, p in enumerate(ranked[:k]):
        if qrels.get(p, 0) >= thr:
            return 1.0 / (r + 1)
    return 0.0


def evaluate(retriever, items: list[dict], chunk_pair: list[str], prefix: str,
             rel_threshold: int = 1, depth: int = 30) -> tuple[dict, list[dict]]:
    """Run every query, return (aggregate metrics, per-query rows)."""
    t0 = time.time()
    all_hits = retriever.search_many([it["query"] for it in items], depth)
    elapsed = time.time() - t0
    rows = []
    for it, hits in zip(items, all_hits):
        ranked = dedupe_pairs(hits, chunk_pair)
        q = it["qrels"]
        rows.append({
            "qid": it["qid"], "query": it["query"],
            "ndcg@10": ndcg_at(ranked, q, 10), "recall@5": recall_at(ranked, q, 5, rel_threshold),
            "recall@10": recall_at(ranked, q, 10, rel_threshold), "hit@5": hit_at(ranked, q, 5, rel_threshold),
            "mrr@10": mrr(ranked, q, rel_threshold), "top_score": hits[0][1] if hits else 0.0,
            "top_pairs": ranked[:5],
        })
    agg = {f"{prefix}/{m}": float(np.mean([r[m] for r in rows])) for m in
           ("ndcg@10", "recall@5", "recall@10", "hit@5", "mrr@10")}
    agg[f"{prefix}/n"] = len(rows)
    agg[f"{prefix}/ms_per_query"] = 1000 * elapsed / max(1, len(items))
    return agg, rows
