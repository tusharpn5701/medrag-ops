"""Register the challenger and decide whether it replaces the champion.

The decision re-evaluates BOTH models on the current held-out test set, so a change in
eval data never compares fresh numbers against stale ones.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ARTIFACTS, DATA_PROCESSED, REPORTS
from .data import read_jsonl
from .retrieval import Bundle
from .text import ci_notice
from .tracking import (
    CODE_PATH,
    MODEL_NAME,
    REQUIREMENTS,
    _pyfunc_class,
    download_bundle,
    get_version_by_role,
    mlflow_ctx,
    set_role,
)
from .train import evaluate_all, load_eval_sets


def _decide(params: dict, chal: dict, champ: dict | None, data_changed: bool = False) -> tuple[bool, list[str]]:
    """Promote when guardrails hold AND the challenger is strictly better on the decision metric.

    Exception: if the corpus changed, a challenger that is no worse is promoted so production
    serves fresh documents.
    """
    p = params["promotion"]
    reasons, ok = [], True
    for metric, floor in p.get("guardrails", {}).items():
        val = chal.get(metric)
        if val is None or val < floor:
            ok = False
            reasons.append(f"guardrail failed: {metric}={val if val is None else round(val, 4)} < {floor}")
    if champ is None:
        reasons.append("no champion yet")
        return ok, reasons
    key = p["metric"]
    delta = chal.get(key, 0.0) - champ.get(key, 0.0)
    summary = f"{key}: challenger {chal.get(key, 0):.4f} vs champion {champ.get(key, 0):.4f} (delta {delta:+.4f})"
    if delta > p.get("min_improvement", 0.0):
        reasons.append(summary + ", improved")
    elif data_changed and delta >= -1e-9:
        reasons.append(summary + ", corpus changed and quality held, refreshing")
    else:
        ok = False
        reasons.append(summary + ", not better")
    return ok, reasons


def register_and_promote(params: dict, bundle_dir: Path = ARTIFACTS / "challenger", log=ci_notice) -> dict:
    chunks_pair = [c["pair_id"] for c in read_jsonl(DATA_PROCESSED / "chunks.jsonl")]
    sets = {k: v for k, v in load_eval_sets().items() if k in ("medquad", "liveqa_test", "prod_val")}
    challenger = Bundle.load(bundle_dir)
    chal_metrics, _ = evaluate_all(challenger.retriever, sets, chunks_pair)

    with mlflow_ctx() as mlflow:
        from mlflow import MlflowClient
        client = MlflowClient()
        mlflow.set_experiment("medrag-retrieval")
        t_up = time.time()
        with mlflow.start_run(run_id=challenger.manifest["run_id"]):
            info = mlflow.pyfunc.log_model(
                artifact_path="model", python_model=_pyfunc_class()(), artifacts={"bundle": str(bundle_dir)},
                code_paths=[str(CODE_PATH)], pip_requirements=REQUIREMENTS,
                registered_model_name=MODEL_NAME)
        version = str(info.registered_model_version)
        log(f"logged and registered v{version} in {time.time() - t_up:.1f}s")
        set_role(client, "challenger", version)
        for k, v in chal_metrics.items():
            if isinstance(v, (int, float)):
                client.set_model_version_tag(MODEL_NAME, version, f"metric.{k.replace('@', '_at_')}", f"{v:.5f}")
        client.set_model_version_tag(MODEL_NAME, version, "retriever", challenger.manifest["name"])

        champ_v = get_version_by_role(client, "champion")
        champ_metrics, data_changed = None, False
        if champ_v is not None and champ_v.version != version:
            champ_bundle = Bundle.load(download_bundle(client, champ_v, ARTIFACTS / "champion_eval"))
            champ_metrics, _ = evaluate_all(champ_bundle.retriever, sets, chunks_pair)
            data_changed = champ_bundle.manifest.get("corpus_sha1") != challenger.manifest.get("corpus_sha1")

        promote, reasons = _decide(params, chal_metrics, champ_metrics, data_changed)
        if promote:
            set_role(client, "champion", version)
        log(("PROMOTED" if promote else "KEPT CHAMPION") + f" (challenger v{version}): " + "; ".join(reasons))

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": MODEL_NAME, "challenger_version": version,
        "challenger_retriever": challenger.manifest["name"],
        "champion_version_before": champ_v.version if champ_v is not None else None,
        "promoted": promote, "reasons": reasons, "decision_metric": params["promotion"]["metric"],
        "challenger_metrics": {k: round(v, 5) for k, v in chal_metrics.items()},
        "champion_metrics": {k: round(v, 5) for k, v in champ_metrics.items()} if champ_metrics else None,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "promotion.json").write_text(json.dumps(report, indent=2))
    return report


def export_bundle(role: str = "champion", dest: Path = ARTIFACTS / "bundle") -> dict:
    """Download the bundle for a registry role (used by the deploy workflow)."""
    with mlflow_ctx():
        from mlflow import MlflowClient
        client = MlflowClient()
        v = get_version_by_role(client, role)
        if v is None:
            raise SystemExit(f"No model version has role '{role}' in {MODEL_NAME}")
        download_bundle(client, v, dest)
    manifest = json.loads((dest / "manifest.json").read_text())
    manifest["model_version"] = f"{MODEL_NAME}/v{v.version}"
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"version": v.version, "retriever": manifest["name"], "path": str(dest)}
