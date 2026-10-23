"""Production monitoring: query drift, retrieval confidence, latency, feedback -> retrain decision.

Query drift uses a domain classifier: train a model to tell reference queries (what the
retriever was evaluated on) from recent production queries. Cross-validated ROC AUC near
0.5 means the two are indistinguishable; high AUC means production questions look
different, and the classifier's top-weighted words show *how* they differ.
"""
from __future__ import annotations

import html
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .config import ARTIFACTS, ROOT
from .logsink import load_logs


# --------------------------------------------------------------------------- statistics
def domain_classifier_auc(reference: list[str], current: list[str], seed: int = 0) -> tuple[float, list, list]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    rng = np.random.default_rng(seed)
    ref = list(rng.choice(reference, size=min(len(reference), max(len(current), 200)), replace=False))
    texts, y = ref + current, np.array([0] * len(ref) + [1] * len(current))
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    folds = min(5, int(min(np.bincount(y))))
    proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    clf.fit(X, y)
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    return auc, [(terms[i], round(float(clf.coef_[0][i]), 3)) for i in order[::-1][:12]], \
        [(terms[i], round(float(clf.coef_[0][i]), 3)) for i in order[:8]]


def _unique_low(traffic: list[dict], n: int = 12) -> list[dict]:
    seen, out = set(), []
    for r in sorted(traffic, key=lambda r: r.get("top_score", 0)):
        if r.get("low_confidence") and r["query"] not in seen:
            seen.add(r["query"])
            out.append({"query": r["query"][:200], "top_score": r.get("top_score")})
        if len(out) == n:
            break
    return out


def _q(values, q):
    return float(np.quantile(values, q)) if len(values) else None


def compute_report(params: dict, logs: list[dict], reference: list[dict], manifest: dict) -> dict:
    m = params["monitoring"]
    traffic = [r for r in logs if r.get("endpoint") in ("ask", "search")]
    feedback = [r for r in logs if r.get("endpoint") == "feedback"]
    n = len(traffic)
    thr = manifest.get("monitoring", {}).get("low_confidence_threshold", 0.0)
    ref_scores = [r["top_score"] for r in reference]
    rep = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window_days": m["window_days"], "model_version": manifest.get("model_version"),
        "retriever": manifest.get("name"), "n_queries": n,
        "n_simulated": sum(1 for r in traffic if r.get("source") == "simulated"),
        "daily": dict(sorted(Counter(r["ts"][:10] for r in traffic).items())),
        "daily_simulated": dict(sorted(Counter(r["ts"][:10] for r in traffic if r.get("source") == "simulated").items())),
        "low_confidence_threshold": thr,
        "reference": {"n": len(reference), "top_score_median": _q(ref_scores, 0.5),
                      "top_scores": ref_scores,
                      "query_words_median": _q([len(r["query"].split()) for r in reference], 0.5)},
    }
    if n:
        lat = [r.get("latency_ms", 0) for r in traffic]
        asks = [r for r in traffic if r.get("endpoint") == "ask"]
        rep.update({
            "top_scores": [r.get("top_score", 0.0) for r in traffic],
            "top_score_median": _q([r.get("top_score", 0.0) for r in traffic], 0.5),
            "low_confidence_rate": float(np.mean([r.get("low_confidence", False) for r in traffic])),
            "latency_p50_ms": _q(lat, 0.5), "latency_p95_ms": _q(lat, 0.95),
            "llm_fallback_rate": float(np.mean([bool(r.get("gen_error")) for r in asks])) if asks else None,
            "llm_share": float(np.mean([r.get("gen_mode") == "llm" for r in asks])) if asks else None,
            "query_words_median": _q([len(r["query"].split()) for r in traffic], 0.5),
            "low_confidence_examples": _unique_low(traffic),
        })
    ups = sum(1 for f in feedback if f.get("rating") == "up")
    downs = sum(1 for f in feedback if f.get("rating") == "down")
    rep["feedback"] = {"up": ups, "down": downs, "down_rate": downs / (ups + downs) if ups + downs else None}

    reasons, retrain, alerts = [], False, []
    if n < m["min_queries"]:
        status = "insufficient_data"
        reasons.append(f"only {n} queries in the last {m['window_days']} days (need {m['min_queries']})")
    else:
        auc, prod_terms, ref_terms = domain_classifier_auc([r["query"] for r in reference], [r["query"] for r in traffic])
        rep["drift"] = {"auc": auc, "threshold": m["drift_auc_threshold"],
                        "production_terms": prod_terms, "reference_terms": ref_terms}
        if auc > m["drift_auc_threshold"]:
            retrain = True
            reasons.append(f"query drift: domain-classifier AUC {auc:.2f} > {m['drift_auc_threshold']}")
        if rep["low_confidence_rate"] > m["max_low_confidence_rate"]:
            retrain = True
            reasons.append(f"low-confidence rate {rep['low_confidence_rate']:.0%} > {m['max_low_confidence_rate']:.0%}")
        if rep["feedback"]["down_rate"] is not None and ups + downs >= 10 and rep["feedback"]["down_rate"] > 0.3:
            retrain = True
            reasons.append(f"thumbs-down rate {rep['feedback']['down_rate']:.0%} on {ups + downs} ratings")
        if rep["latency_p95_ms"] > m["max_p95_latency_ms"]:
            alerts.append(f"p95 latency {rep['latency_p95_ms']:.0f} ms > {m['max_p95_latency_ms']} ms")
        if rep.get("llm_fallback_rate") and rep["llm_fallback_rate"] > m["max_error_rate"]:
            alerts.append(f"LLM fallback rate {rep['llm_fallback_rate']:.0%} (provider errors or rate limits)")
        status = "retrain" if retrain else ("alert" if alerts else "healthy")
        if not reasons:
            reasons.append("query mix and retrieval confidence are in line with the reference set")
    rep.update({"status": status, "retrain_recommended": retrain, "reasons": reasons, "alerts": alerts})
    return rep


# --------------------------------------------------------------------------- entry point
def run_monitor(params: dict, logs_dir: str | None, out_dir: Path) -> dict:
    bundle = Path(os.environ.get("MEDRAG_BUNDLE_DIR", ARTIFACTS / "bundle"))
    manifest = json.loads((bundle / "manifest.json").read_text())
    with open(bundle / "reference_queries.jsonl") as f:
        reference = [json.loads(line) for line in f]
    logs = load_logs(params["monitoring"]["window_days"], local_dir=logs_dir or (None if os.environ.get("MEDRAG_LOG_BUCKET") else ROOT / "logs"),
                     bucket=None if logs_dir else os.environ.get("MEDRAG_LOG_BUCKET"),
                     prefix=os.environ.get("MEDRAG_LOG_PREFIX", "request-logs"),
                     endpoint_url=os.environ.get("MEDRAG_S3_ENDPOINT_URL"))
    rep = compute_report(params, logs, reference, manifest)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    history.append({k: rep.get(k) for k in ("generated_at", "status", "n_queries", "low_confidence_rate",
                                            "latency_p95_ms", "model_version")} | {"drift_auc": rep.get("drift", {}).get("auc")})
    history = history[-90:]
    hist_path.write_text(json.dumps(history, indent=1))
    promotion = ROOT / "reports" / "promotion.json"
    promo = json.loads(promotion.read_text()) if promotion.exists() else None
    slim = {k: v for k, v in rep.items() if k not in ("top_scores",)}
    slim["reference"] = {k: v for k, v in rep["reference"].items() if k != "top_scores"}
    (out_dir / "report.json").write_text(json.dumps(slim, indent=2))
    (out_dir / "index.html").write_text(render_dashboard(rep, history, promo))
    return rep


# --------------------------------------------------------------------------- dashboard
def _svg_bars(daily: dict, simulated: dict, w=560, h=170) -> str:
    if not daily:
        return '<p class="muted">No traffic in this window yet.</p>'
    days = list(daily)
    top = max(daily.values())
    step = 10 ** int(np.floor(np.log10(max(top, 1))))
    ymax = int(np.ceil(top / step) * step) or 1
    ml, mb, mt = 36, 26, 8
    bw = min((w - ml - 8) / len(days), 48)
    y = lambda v: mt + (h - mt - mb) * (1 - v / ymax)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Queries per day">']
    for t in np.linspace(0, ymax, 3):
        out.append(f'<line x1="{ml}" x2="{w - 8}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>'
                   f'<text x="{ml - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{int(t)}</text>')
    for i, d in enumerate(days):
        x, total, sim = ml + i * bw + 1, daily[d], simulated.get(d, 0)
        out.append(f'<rect x="{x:.1f}" y="{y(total):.1f}" width="{max(bw - 2, 1):.1f}" height="{y(0) - y(total):.1f}" class="b-user" rx="2"><title>{d}: {total} queries</title></rect>')
        if sim:
            out.append(f'<rect x="{x:.1f}" y="{y(sim):.1f}" width="{max(bw - 2, 1):.1f}" height="{y(0) - y(sim):.1f}" class="b-sim" rx="2"><title>{d}: {sim} simulated</title></rect>')
        if len(days) <= 14 or i % max(1, len(days) // 7) == 0:
            out.append(f'<text x="{x + bw / 2:.1f}" y="{h - 8}" text-anchor="middle">{d[5:]}</text>')
    return "".join(out) + "</svg>"


def _svg_hist(ref: list[float], cur: list[float], thr: float, w=560, h=170) -> str:  # noqa: C901
    if not ref:
        return ""
    hi = float(np.quantile(ref + cur, 0.99)) or 1.0
    bins = np.linspace(0, hi, 25)
    rh = np.histogram(np.clip(ref, 0, hi), bins)[0] / max(len(ref), 1)
    ch = np.histogram(np.clip(cur, 0, hi), bins)[0] / max(len(cur), 1) if cur else np.zeros(len(bins) - 1)
    ymax = max(rh.max(), ch.max()) * 1.1 or 1
    ml, mb, mt = 36, 26, 8
    x = lambda v: ml + (w - ml - 24) * v / hi  # noqa: E731
    y = lambda v: mt + (h - mt - mb) * (1 - v / ymax)  # noqa: E731

    def path(vals):
        pts = [f"{x(bins[i]):.1f},{y(v):.1f} {x(bins[i + 1]):.1f},{y(v):.1f}" for i, v in enumerate(vals)]
        return "M" + f"{x(0):.1f},{y(0):.1f} " + " ".join(pts) + f" {x(hi):.1f},{y(0):.1f}"
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Top retrieval score distribution">',
           f'<line x1="{ml}" x2="{w - 8}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="grid"/>',
           f'<path d="{path(rh)}" class="h-ref"/>']
    if cur:
        out.append(f'<path d="{path(ch)}" class="h-cur"/>')
    tx = x(min(thr, hi))
    out.append(f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{mt}" y2="{y(0):.1f}" class="thr"/>'
               f'<text x="{tx + 4:.1f}" y="{mt + 10}">low-confidence threshold</text>')
    for v in np.linspace(0, hi, 5):
        out.append(f'<text x="{x(v):.1f}" y="{h - 8}" text-anchor="middle">{v:.1f}</text>')
    return "".join(out) + "</svg>"


def _fmt2(v) -> str:
    return "–" if v is None else f"{v:.2f}"


def render_dashboard(rep: dict, history: list[dict], promo: dict | None) -> str:
    e = html.escape
    status_text = {"healthy": "Healthy", "retrain": "Retraining triggered", "alert": "Needs attention",
                   "insufficient_data": "Waiting for traffic"}[rep["status"]]

    def pct(v):
        return "–" if v is None else f"{100 * v:.0f}%"

    def num(v, unit=""):
        return "–" if v is None else f"{v:,.0f}{unit}"
    kpis = [("Queries", num(rep["n_queries"]), f"{rep['n_simulated']} simulated"),
            ("Low confidence", pct(rep.get("low_confidence_rate")), f"best hit under {rep['low_confidence_threshold']:.2f}"),
            ("Drift AUC", "–" if "drift" not in rep else f"{rep['drift']['auc']:.2f}", f"alert above {rep['drift']['threshold'] if 'drift' in rep else '0.75'}"),
            ("p95 latency", num(rep.get("latency_p95_ms"), " ms"), f"p50 {num(rep.get('latency_p50_ms'), ' ms')}"),
            ("LLM answers", pct(rep.get("llm_share")), f"fallbacks {pct(rep.get('llm_fallback_rate'))}"),
            ("Thumbs down", pct(rep["feedback"]["down_rate"]), f"{rep['feedback']['up'] + rep['feedback']['down']} ratings")]
    terms = ""
    if "drift" in rep:
        terms = ('<div class="terms"><div><h3>More common in production</h3><ul>' +
                 "".join(f"<li>{e(t)}</li>" for t, _ in rep["drift"]["production_terms"]) +
                 '</ul></div><div><h3>More common in the reference set</h3><ul>' +
                 "".join(f"<li>{e(t)}</li>" for t, _ in rep["drift"]["reference_terms"]) + "</ul></div></div>")
    lows = "".join(f"<tr><td>{e(x['query'])}</td><td class='n'>{x['top_score']:.2f}</td></tr>"
                   for x in rep.get("low_confidence_examples", []))
    hist_rows = "".join(
        f"<tr><td>{e(h['generated_at'][:16].replace('T', ' '))}</td><td>{e(h['status'])}</td><td class='n'>{h['n_queries']}</td>"
        f"<td class='n'>{_fmt2(h.get('drift_auc'))}</td>"
        f"<td class='n'>{pct(h.get('low_confidence_rate'))}</td><td>{e(str(h.get('model_version') or '–'))}</td></tr>"
        for h in reversed(history[-12:]))
    promo_html = ""
    if promo:
        promo_html = (f"<p><b>Last training run</b> ({e(promo['timestamp'][:10])}): challenger "
                      f"v{e(str(promo['challenger_version']))} ({e(promo['challenger_retriever'])}) was "
                      f"{'promoted' if promo['promoted'] else 'not promoted'}. {e('; '.join(promo['reasons']))}</p>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>MedRAG Ops Monitoring</title>
<style>
:root {{ --bg:#f4f6f7; --surface:#fff; --ink:#14212b; --ink-2:#4b5b66; --ink-3:#7a8891; --line:#dbe2e6;
  --ref:#8a9aa5; --cur:#0f6e8c; --sim:#b9d7e1; --good:#1e7a46; --warn:#a15c00; --bad:#b42318; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f171c; --surface:#16222a; --ink:#e5edf1; --ink-2:#a9b8c1;
  --ink-3:#7d8d97; --line:#26353f; --ref:#6d7f8a; --cur:#4fb3d4; --sim:#2b5a6a; --good:#5cc489; --warn:#f0b45c; --bad:#f2786c; }} }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif }}
.wrap {{ max-width:1080px; margin:0 auto; padding:28px max(16px,env(safe-area-inset-left)) 48px; display:grid; gap:18px }}
h1 {{ margin:0; font-size:26px }} h2 {{ margin:0 0 8px; font-size:17px }} h3 {{ margin:0 0 6px; font-size:14px; color:var(--ink-2) }}
.muted, .sub {{ color:var(--ink-3) }} .status {{ padding:12px 16px; border-radius:10px; background:var(--surface); border:1px solid var(--line); border-left:6px solid var(--st) }}
.status b {{ color:var(--st) }} .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px }}
.kpi {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 14px }}
.kpi .v {{ font-size:26px; font-weight:650; font-variant-numeric:tabular-nums }} .kpi .l {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-3) }}
.kpi .s {{ font-size:13px; color:var(--ink-3) }} .grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px; min-width:0 }}
svg {{ width:100%; height:auto; display:block }} svg text {{ fill:var(--ink-3); font:11px ui-monospace,monospace }}
.grid {{ stroke:var(--line) }} .b-user {{ fill:var(--cur) }} .b-sim {{ fill:var(--sim) }}
.h-ref {{ fill:var(--ref); fill-opacity:.25; stroke:var(--ref); stroke-width:1.5 }} .h-cur {{ fill:var(--cur); fill-opacity:.25; stroke:var(--cur); stroke-width:2 }}
.thr {{ stroke:var(--warn); stroke-dasharray:4 4 }} .legend {{ display:flex; gap:14px; font-size:13px; color:var(--ink-2); flex-wrap:wrap }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px }}
.terms {{ display:grid; grid-template-columns:1fr 1fr; gap:12px }} .terms ul {{ margin:0; padding-left:18px }}
.tw {{ overflow-x:auto }} table {{ border-collapse:collapse; width:100% }} td,th {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); font-size:14px }}
th {{ color:var(--ink-3); font-weight:500; font-size:12px; text-transform:uppercase; letter-spacing:.05em }} .n {{ text-align:right; font-variant-numeric:tabular-nums }}
a {{ color:var(--cur) }}
</style></head><body><div class="wrap">
<div><h1>MedRAG Ops monitoring</h1><p class="sub">Last {rep['window_days']} days · generated {e(rep['generated_at'])} · serving {e(str(rep['model_version']))} ({e(str(rep['retriever']))}) · <a href="report.json">report.json</a> · <a href="https://github.com/tusharpn5701/medrag-ops">code</a></p></div>
<div class="status" style="--st:var({'--good' if rep['status'] == 'healthy' else '--bad' if rep['status'] == 'retrain' else '--warn'})"><b>{status_text}.</b> {e('; '.join(rep['reasons'] + rep['alerts']))}</div>
<div class="kpis">{''.join(f'<div class="kpi"><div class="l">{a}</div><div class="v">{b}</div><div class="s">{e(c)}</div></div>' for a, b, c in kpis)}</div>
<div class="grid2">
<div class="card"><h2>Queries per day</h2>{_svg_bars(rep['daily'], rep['daily_simulated'])}<div class="legend"><span><i class="sw" style="background:var(--cur)"></i>all traffic</span><span><i class="sw" style="background:var(--sim)"></i>simulated share</span></div></div>
<div class="card"><h2>Best-hit retrieval score</h2>{_svg_hist(rep['reference']['top_scores'], rep.get('top_scores', []), rep['low_confidence_threshold'])}<div class="legend"><span><i class="sw" style="background:var(--ref)"></i>reference queries ({rep['reference']['n']})</span><span><i class="sw" style="background:var(--cur)"></i>production</span></div></div>
</div>
<div class="grid2">
<div class="card"><h2>What changed in the questions</h2>{terms or '<p class="muted">Drift analysis runs once there are enough queries.</p>'}<p class="muted">Median question length: reference {num(rep['reference']['query_words_median'])} words, production {num(rep.get('query_words_median'))} words.</p></div>
<div class="card"><h2>Least confident queries</h2>{'<div class="tw"><table><tr><th>Query (redacted)</th><th class="n">Score</th></tr>' + lows + '</table></div>' if lows else '<p class="muted">None in this window.</p>'}</div>
</div>
<div class="card"><h2>Run history</h2>{promo_html}<div class="tw"><table><tr><th>Run</th><th>Status</th><th class="n">Queries</th><th class="n">Drift AUC</th><th class="n">Low conf.</th><th>Model</th></tr>{hist_rows}</table></div></div>
</div></body></html>"""
