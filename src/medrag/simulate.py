"""Send realistic questions to a running API so the monitoring loop has traffic to analyse.

Requests carry the admin token and an `x-traffic-source: simulated` header, so they skip
the rate limit and are labelled as simulated in logs and on the dashboard.
"""
from __future__ import annotations

import os
import random
import time
import urllib.request
import xml.etree.ElementTree as ET

from .config import DATA_PROCESSED, DATA_RAW
from .data import LIVEQA_TEST_XML, read_jsonl
from .text import normalize_ws

OFF_TOPIC = [
    "how do I renew my passport online", "best budget laptop for programming 2026",
    "what is the capital of australia", "how to fix a leaking kitchen tap",
    "can my landlord increase rent twice a year", "recipe for vegan chocolate cake",
]


def _liveqa_questions() -> list[str]:
    path = DATA_RAW / "liveqa" / "test_questions.xml"
    raw = path.read_bytes() if path.exists() else urllib.request.urlopen(LIVEQA_TEST_XML, timeout=60).read()
    out = []
    for q in ET.fromstring(raw).iter("NLM-QUESTION"):
        o = q.find("Original-Question")
        if o is not None:
            out.append(normalize_ws(f"{o.findtext('SUBJECT') or ''}. {o.findtext('MESSAGE') or ''}").strip(". "))
    return [q for q in out if len(q) > 10]


def simulate(url: str, n: int = 60, mix: str = "liveqa", seed: int | None = None) -> dict:
    import json
    rng = random.Random(seed)
    pool = []
    if mix in ("liveqa", "mixed"):
        pool += _liveqa_questions()
    if mix in ("medquad", "mixed") and (DATA_PROCESSED / "eval_medquad.jsonl").exists():
        pool += [r["query"] for r in read_jsonl(DATA_PROCESSED / "eval_medquad.jsonl")]
    if mix == "mixed":
        pool += OFF_TOPIC * 3
    headers = {"Content-Type": "application/json", "x-traffic-source": "simulated",
               "x-admin-token": os.environ.get("MEDRAG_ADMIN_TOKEN", "")}
    ok = errors = 0
    lat = []
    for _ in range(n):
        q = rng.choice(pool)[:900]
        req = urllib.request.Request(url.rstrip("/") + "/ask", data=json.dumps({"query": q}).encode(), headers=headers)
        t0 = time.time()
        try:
            urllib.request.urlopen(req, timeout=60).read()
            ok += 1
            lat.append(time.time() - t0)
        except Exception:
            errors += 1
        time.sleep(0.2)
    return {"sent": n, "ok": ok, "errors": errors, "mean_latency_s": round(sum(lat) / max(len(lat), 1), 3)}
