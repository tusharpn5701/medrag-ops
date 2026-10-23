"""Turn production questions into evaluation data with an LLM judge.

For recent queries (low-confidence ones first) the champion retrieves its top passages
and the LLM grades each one 0-3 using the same scale as the NIH LiveQA judgements.
Queries with at least one passage graded >= 2 join eval_production.jsonl, which the next
sweep uses for model selection and as monitoring reference. Queries where nothing is
relevant are counted as corpus gaps.

Caveat: passages come from the current champion only (single-system pooling), so these
labels favour the champion. The NIH-judged LiveQA test split remains the promotion gate.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

from .config import ARTIFACTS, FEEDBACK_FILE, get_settings
from .data import read_jsonl, write_jsonl
from .logsink import _s3, load_logs
from .retrieval import Bundle
from .text import norm_key, sha1

JUDGE_PROMPT = """You grade search results for a consumer health question.
Scale: 3 = fully answers it, 2 = answers part of it, 1 = on the topic but does not answer, 0 = not relevant.
Question: {question}

Passages:
{passages}

Reply with JSON only, like {{"grades": [3, 0, 1]}}, one grade per passage in order."""

EVAL_KEY = "eval/eval_production.jsonl"


def _bucket_io(download: bool, path: Path) -> None:
    bucket = os.environ.get("MEDRAG_LOG_BUCKET")
    if not bucket:
        return
    s3 = _s3(os.environ.get("MEDRAG_S3_ENDPOINT_URL"))
    try:
        if download:
            path.write_bytes(s3.get_object(Bucket=bucket, Key=EVAL_KEY)["Body"].read())
        else:
            s3.put_object(Bucket=bucket, Key=EVAL_KEY, Body=path.read_bytes())
    except Exception as e:  # first run: nothing stored yet
        print(f"  bucket {'download' if download else 'upload'} skipped: {type(e).__name__}")


def parse_grades(text: str, n: int) -> list[int] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        grades = json.loads(m.group(0))["grades"] if m else None
    except (ValueError, KeyError):
        grades = None
    if not isinstance(grades, list) or len(grades) != n:
        return None
    return [max(0, min(3, int(g))) for g in grades]


def label_production_queries(params: dict, logs_dir: str | None, limit: int = 40, client=None) -> dict:
    s = get_settings()
    if client is None:
        if not s.llm_api_key:
            return {"skipped": "no LLM API key (set GROQ_API_KEY)"}
        from openai import OpenAI
        client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url, timeout=30, max_retries=2)
    out_path = FEEDBACK_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _bucket_io(True, out_path)
    existing = read_jsonl(out_path) if out_path.exists() else []
    seen = {norm_key(r["query"]) for r in existing}

    logs = load_logs(params["monitoring"]["window_days"], local_dir=logs_dir,
                     bucket=None if logs_dir else os.environ.get("MEDRAG_LOG_BUCKET"),
                     prefix=os.environ.get("MEDRAG_LOG_PREFIX", "request-logs"),
                     endpoint_url=os.environ.get("MEDRAG_S3_ENDPOINT_URL"))
    pool, keys = [], set()
    for r in sorted((r for r in logs if r.get("endpoint") in ("ask", "search")),
                    key=lambda r: (not r.get("low_confidence"), random.random())):
        k = norm_key(r["query"])
        if k and k not in seen and k not in keys:
            keys.add(k)
            pool.append(r["query"])
    pool = pool[:limit]

    bundle = Bundle.load(Path(os.environ.get("MEDRAG_BUNDLE_DIR", ARTIFACTS / "bundle")))
    added, gaps, failed = [], 0, 0
    for q in pool:
        hits, pairs = [], set()
        for h in bundle.search(q, 20):
            if h["pair_id"] not in pairs:
                pairs.add(h["pair_id"])
                hits.append(h)
            if len(hits) == 8:
                break
        passages = "\n".join(f"[{i + 1}] {h['focus']}: {' '.join(h['body'].split()[:120])}" for i, h in enumerate(hits))
        try:
            resp = client.chat.completions.create(
                model=params["generation"]["model"], temperature=0, max_tokens=200,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=q, passages=passages)}])
            grades = parse_grades(resp.choices[0].message.content, len(hits))
        except Exception:
            grades = None
        if grades is None:
            failed += 1
            continue
        qrels = {h["pair_id"]: g for h, g in zip(hits, grades) if g > 0}
        if max(grades, default=0) >= 2:
            added.append({"qid": f"prod-{sha1(q)[:10]}", "query": q, "qrels": qrels,
                          "labelled_by": params["generation"]["model"], "labelled_at": time.strftime("%Y-%m-%d")})
        else:
            gaps += 1
        time.sleep(2.2)  # stay under free-tier rate limits
    write_jsonl(out_path, existing + added)
    _bucket_io(False, out_path)
    return {"candidates": len(pool), "added": len(added), "corpus_gaps": gaps, "judge_failures": failed,
            "total_labelled": len(existing) + len(added)}
