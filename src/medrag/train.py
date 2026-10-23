"""Candidate sweep: build every retriever in params.yaml, evaluate, log to MLflow, pick the challenger.

Selection uses the LiveQA *validation* split (plus production queries labelled by the
feedback loop, when present). The held-out LiveQA *test* split is only used later, in
registry.promote(), so the promotion decision is not made on data used for selection.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from .config import ARTIFACTS, DATA_PROCESSED, FEEDBACK_FILE, REPORTS
from .data import read_jsonl
from .evaluate import evaluate
from .retrieval import Bundle, HybridRetriever, build_retrievers
from .text import ci_notice
from .tracking import flatten, mlflow_ctx

EVAL_SETS = {  # name -> (file, relevance threshold for recall/MRR)
    "medquad": ("eval_medquad.jsonl", 1),
    "liveqa_val": ("eval_liveqa_val.jsonl", 2),
    "liveqa_test": ("eval_liveqa_test.jsonl", 2),
    "prod_val": ("eval_production.jsonl", 2),   # optional: produced by the feedback loop
}


def load_eval_sets(data_dir: Path = DATA_PROCESSED) -> dict[str, tuple[list[dict], int]]:
    sets = {}
    for name, (fname, thr) in EVAL_SETS.items():
        p = FEEDBACK_FILE if name == "prod_val" else data_dir / fname
        if p.exists():
            rows = read_jsonl(p)
            if rows:
                sets[name] = (rows, thr)
    return sets


def evaluate_all(retriever, sets, chunk_pair) -> tuple[dict, dict]:
    metrics, per_query = {}, {}
    for name, (items, thr) in sets.items():
        m, rows = evaluate(retriever, items, chunk_pair, name, rel_threshold=thr)
        metrics.update(m)
        per_query[name] = rows
    return metrics, per_query


def reference_profile(retriever, sets, params) -> dict:
    """Top-score distribution on the eval queries -> thresholds the API and monitor use."""
    import numpy as np
    rows = []
    for name in ("medquad", "liveqa_val", "prod_val"):
        if name not in sets:
            continue
        items = sets[name][0]
        hits = retriever.search_many([it["query"] for it in items], 1)
        rows += [{"query": it["query"], "set": name, "top_score": h[0][1] if h else 0.0}
                 for it, h in zip(items, hits)]
    scores = np.array([r["top_score"] for r in rows]) if rows else np.array([0.0])
    q = params["monitoring"]["low_confidence_quantile"]
    return {"rows": rows, "thresholds": {
        "low_confidence_threshold": float(np.quantile(scores, q)),
        "reference_top_score_median": float(np.median(scores)),
        "reference_query_words_median": float(np.median([len(r["query"].split()) for r in rows] or [0]))}}


def selection_score(metrics: dict, params: dict) -> float:
    key = params["selection"]["metric"]
    score = metrics.get(key, 0.0)
    if "prod_val/ndcg@10" in metrics:  # production-labelled queries count equally with LiveQA val
        score = 0.5 * score + 0.5 * metrics["prod_val/ndcg@10"]
    return score


def run_sweep(params: dict, only: list[str] | None = None, out_dir: Path = ARTIFACTS, log=ci_notice) -> dict:
    strict = os.environ.get("MEDRAG_STRICT") == "1"
    chunks = read_jsonl(DATA_PROCESSED / "chunks.jsonl")
    stats = json.loads((DATA_PROCESSED / "stats.json").read_text())
    texts = [c["text"] for c in chunks]
    chunk_pair = [c["pair_id"] for c in chunks]
    sets = load_eval_sets()
    candidates = [c for c in params["retrieval"]["candidates"] if not only or c["name"] in only]

    # Fit base retrievers one by one so a missing optional dependency skips only that candidate.
    fitted, skipped = {}, {}
    for cand in [c for c in candidates if c["kind"] != "hybrid"]:
        try:
            fitted.update(build_retrievers([cand], texts, params["retrieval"]["candidate_depth"], log))
        except Exception as e:
            if strict:
                raise
            skipped[cand["name"]] = f"{type(e).__name__}: {str(e)[:160]}"
            log(f"  skipped {cand['name']}: {skipped[cand['name']]}")
    for cand in [c for c in candidates if c["kind"] == "hybrid"]:
        missing = [m for m in cand["members"] if m not in fitted]
        if missing:
            skipped[cand["name"]] = f"members not available: {missing}"
            continue
        h = HybridRetriever(cand["name"], {**cand, "candidate_depth": params["retrieval"]["candidate_depth"]},
                            [fitted[m] for m in cand["members"]])
        h.fit_seconds = sum(fitted[m].fit_seconds for m in cand["members"])
        fitted[cand["name"]] = h

    results = []
    with mlflow_ctx() as mlflow:
        mlflow.set_experiment("medrag-retrieval")
        with mlflow.start_run(run_name=f"sweep-{time.strftime('%Y%m%d-%H%M%S')}") as parent:
            mlflow.log_params(flatten({"chunking": params["chunking"], "data": params["data"]}))
            mlflow.log_metrics({f"data/{k}": v for k, v in stats.items() if isinstance(v, (int, float))})
            mlflow.set_tag("corpus_sha1", stats["corpus_sha1"])
            if skipped:
                mlflow.set_tag("skipped_candidates", json.dumps(skipped))
            for name, retriever in fitted.items():
                t_eval = time.time()
                metrics, per_query = evaluate_all(retriever, sets, chunk_pair)
                log(f"evaluated {name} in {time.time() - t_eval:.1f}s")
                metrics["fit_seconds"] = retriever.fit_seconds
                score = selection_score(metrics, params)
                with mlflow.start_run(run_name=name, nested=True) as run:
                    mlflow.log_params(flatten({"retriever": retriever.config}))
                    mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})
                    mlflow.log_metric("selection_score", score)
                    mlflow.log_dict(per_query, "per_query.json")
                results.append({"name": name, "kind": retriever.kind, "run_id": run.info.run_id,
                                "selection_score": score, "metrics": metrics})
                log(f"  {name:<18} selection={score:.4f}  " + "  ".join(
                    f"{k}={metrics[k]:.3f}" for k in ("medquad/recall@5", "liveqa_val/ndcg@10",
                                                       "liveqa_test/ndcg@10") if k in metrics))
            results.sort(key=lambda r: (r["selection_score"], r["metrics"].get("medquad/recall@5", 0)),
                         reverse=True)
            best = results[0]
            mlflow.set_tag("challenger", best["name"])
            mlflow.log_dict({"candidates": results, "skipped": skipped}, "candidates.json")

            bundle_dir = out_dir / "challenger"
            shutil.rmtree(bundle_dir, ignore_errors=True)
            reference = reference_profile(fitted[best["name"]], sets, params)
            manifest = {"name": best["name"], "run_id": best["run_id"], "sweep_run_id": parent.info.run_id,
                        "corpus_sha1": stats["corpus_sha1"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "metrics": best["metrics"], "params": {"chunking": params["chunking"]},
                        "monitoring": reference["thresholds"]}
            t_save = time.time()
            Bundle(manifest, chunks, fitted[best["name"]]).save(bundle_dir)
            log(f"saved challenger bundle ({best['name']}) in {time.time() - t_save:.1f}s")
            # Reference queries travel with the model so monitoring can compare production against them
            with open(bundle_dir / "reference_queries.jsonl", "w") as f:
                for row in reference["rows"]:
                    f.write(json.dumps(row) + "\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {"challenger": best["name"], "selection_metric": params["selection"]["metric"],
               "candidates": [{k: v for k, v in r.items() if k != "run_id"} for r in results],
               "skipped": skipped}
    (REPORTS / "candidates.json").write_text(json.dumps(summary, indent=2))
    (REPORTS / "challenger_metrics.json").write_text(json.dumps(
        {k: round(v, 5) for k, v in best["metrics"].items()}, indent=2))
    return summary
