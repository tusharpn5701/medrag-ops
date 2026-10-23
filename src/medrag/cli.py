"""Command line entry point: `medrag <command>` (or `python -m medrag.cli <command>`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ARTIFACTS, load_params


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="medrag")
    p.add_argument("--params", default=None, help="path to params.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="fetch MedQuAD and the LiveQA test collection")
    sub.add_parser("prepare", help="build chunks and evaluation sets")
    t = sub.add_parser("train", help="fit + evaluate every candidate retriever, pick the challenger")
    t.add_argument("--only", nargs="*", help="restrict to these candidate names")
    sub.add_parser("register", help="register the challenger and run the champion/challenger gate")
    e = sub.add_parser("export", help="download a registry role's bundle for serving")
    e.add_argument("--role", default="champion")
    e.add_argument("--out", default=str(ARTIFACTS / "bundle"))
    s = sub.add_parser("serve", help="run the API locally")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    m = sub.add_parser("monitor", help="drift + health report from request logs")
    m.add_argument("--logs", default=None, help="local folder of JSONL logs (default: S3 bucket or ./logs)")
    m.add_argument("--out", default="docs/monitoring")
    lab = sub.add_parser("label", help="LLM-judge recent low-confidence queries into eval_production.jsonl")
    lab.add_argument("--logs", default=None)
    lab.add_argument("--limit", type=int, default=40)
    g = sub.add_parser("gate", help="fail if the challenger's metrics drop below reports/baseline.json")
    g.add_argument("--baseline", default="reports/baseline.json")
    sub.add_parser("summary", help="markdown summary of the last sweep/promotion (for CI job summaries)")
    sim = sub.add_parser("simulate", help="send realistic questions to a running API")
    sim.add_argument("--url", required=True)
    sim.add_argument("--n", type=int, default=60)
    sim.add_argument("--mix", default="liveqa", choices=["liveqa", "medquad", "mixed"])
    a = p.parse_args(argv)
    params = load_params(a.params)

    if a.cmd == "download":
        from .data import download
        download()
    elif a.cmd == "prepare":
        from .data import prepare
        print(json.dumps(prepare(params), indent=2))
    elif a.cmd == "train":
        from .train import run_sweep
        out = run_sweep(params, a.only)
        print(f"challenger: {out['challenger']}")
    elif a.cmd == "register":
        from .registry import register_and_promote
        r = register_and_promote(params)
        print(json.dumps({k: r[k] for k in ("challenger_version", "promoted", "reasons")}, indent=2))
    elif a.cmd == "export":
        from .registry import export_bundle
        print(json.dumps(export_bundle(a.role, Path(a.out)), indent=2))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("medrag.api:app", host=a.host, port=a.port)
    elif a.cmd == "monitor":
        from .monitor import run_monitor
        rep = run_monitor(params, a.logs, Path(a.out))
        print(json.dumps({k: rep[k] for k in ("status", "retrain_recommended", "reasons")}, indent=2))
    elif a.cmd == "label":
        from .feedback import label_production_queries
        print(json.dumps(label_production_queries(params, a.logs, a.limit), indent=2))
    elif a.cmd == "gate":
        return quality_gate(Path(a.baseline))
    elif a.cmd == "summary":
        print(markdown_summary())
    elif a.cmd == "simulate":
        from .simulate import simulate
        print(json.dumps(simulate(a.url, a.n, a.mix), indent=2))
    return 0


def quality_gate(baseline_path: Path) -> int:
    from .config import REPORTS
    base = json.loads(baseline_path.read_text())
    got = json.loads((REPORTS / "challenger_metrics.json").read_text())
    tol, failed = base.get("tolerance", 0.0), []
    print(f"{'metric':<24}{'baseline':>10}{'current':>10}")
    for k, floor in base["metrics"].items():
        cur = got.get(k)
        ok = cur is not None and cur >= floor - tol
        print(f"{k:<24}{floor:>10.4f}{(cur if cur is not None else float('nan')):>10.4f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failed.append(k)
    if failed:
        print(f"Quality gate failed on: {', '.join(failed)}")
        return 1
    print("Quality gate passed")
    return 0


def markdown_summary() -> str:
    from .config import REPORTS
    lines = []
    cand = REPORTS / "candidates.json"
    if cand.exists():
        c = json.loads(cand.read_text())
        cols = ["medquad/recall@5", "liveqa_val/ndcg@10", "liveqa_test/ndcg@10", "fit_seconds"]
        lines += [f"### Candidate sweep (selection metric: `{c['selection_metric']}`)", "",
                  "| retriever | selection | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
        for r in c["candidates"]:
            mark = " **(challenger)**" if r["name"] == c["challenger"] else ""
            lines.append(f"| {r['name']}{mark} | {r['selection_score']:.4f} | " +
                         " | ".join(f"{r['metrics'].get(k, float('nan')):.3f}" for k in cols) + " |")
        for name, why in c.get("skipped", {}).items():
            lines.append(f"\nSkipped `{name}`: {why}")
    promo = REPORTS / "promotion.json"
    if promo.exists():
        p = json.loads(promo.read_text())
        verdict = "promoted to champion" if p["promoted"] else "not promoted"
        lines += ["", f"### Registry: challenger v{p['challenger_version']} ({p['challenger_retriever']}) {verdict}", ""]
        lines += [f"- {r}" for r in p["reasons"]]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
