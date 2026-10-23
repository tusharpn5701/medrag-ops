"""`medrag doctor`: check every external service the pipeline depends on, with timings.

Each check prints one line (and a GitHub annotation in CI) so a misconfigured secret or a
slow service shows up in seconds instead of as a timed-out training run.
"""
from __future__ import annotations

import os
import time
import uuid

from .text import ci_notice


def _check(name: str, fn) -> bool:
    t0 = time.time()
    try:
        detail = fn()
        ci_notice(f"OK   {name} ({time.time() - t0:.1f}s){': ' + detail if detail else ''}", title="doctor")
        return True
    except Exception as e:
        ci_notice(f"FAIL {name} ({time.time() - t0:.1f}s): {type(e).__name__}: {str(e)[:300]}", title="doctor")
        return False


def run_doctor(embed_sample: int = 500) -> int:
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
    results = []

    def mlflow_tracking():
        import mlflow
        from mlflow import MlflowClient
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            raise RuntimeError("MLFLOW_TRACKING_URI is not set")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("medrag-doctor")
        with mlflow.start_run(run_name="doctor") as run:
            mlflow.log_param("check", "doctor")
            mlflow.log_metric("ok", 1.0)
            mlflow.log_dict({"hello": "world"}, "doctor.json")
        MlflowClient().search_registered_models(max_results=5)
        return f"run {run.info.run_id[:8]} logged, registry reachable"

    def mlflow_registry_roles():
        from mlflow import MlflowClient

        from .tracking import MODEL_NAME, get_version_by_role
        c = MlflowClient()
        champ = get_version_by_role(c, "champion")
        return f"{MODEL_NAME} champion = {'v' + champ.version if champ else 'none yet'}"

    def bucket():
        from .logsink import _s3
        b = os.environ.get("MEDRAG_LOG_BUCKET")
        if not b:
            raise RuntimeError("MEDRAG_LOG_BUCKET is not set")
        s3 = _s3(os.environ.get("MEDRAG_S3_ENDPOINT_URL"))
        key = f"doctor/{uuid.uuid4().hex}.txt"
        s3.put_object(Bucket=b, Key=key, Body=b"ok")
        body = s3.get_object(Bucket=b, Key=key)["Body"].read()
        try:
            s3.delete_object(Bucket=b, Key=key)
        except Exception:
            pass
        assert body == b"ok"
        return f"put/get on bucket '{b}'"

    def groq():
        from openai import OpenAI

        from .config import load_params
        key = os.environ.get("GROQ_API_KEY") or os.environ.get("MEDRAG_LLM_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        p = load_params()["generation"]
        r = OpenAI(api_key=key, base_url=p["base_url"], timeout=30).chat.completions.create(
            model=p["model"], max_tokens=20, messages=[{"role": "user", "content": "Reply with the word ready."}])
        return f"{p['model']} replied: {(r.choices[0].message.content or '').strip()[:40]!r}"

    def embedding_speed():
        from fastembed import TextEmbedding
        texts = [f"Passage {i}: kidney stones form when urine contains more crystal-forming substances "
                 f"such as calcium oxalate and uric acid than the fluid can dilute. " * 4 for i in range(embed_sample)]
        m = TextEmbedding("BAAI/bge-small-en-v1.5")
        t0 = time.time()
        list(m.passage_embed(texts, batch_size=64))
        rate = embed_sample / (time.time() - t0)
        return f"{rate:.0f} passages/s, so about {28844 / rate / 60:.1f} min for the full corpus"

    results.append(_check("MLflow tracking (DagsHub)", mlflow_tracking))
    if results[-1]:
        results.append(_check("MLflow registry roles", mlflow_registry_roles))
    results.append(_check("Log bucket (S3-compatible)", bucket))
    results.append(_check("Groq LLM", groq))
    results.append(_check("Embedding speed (bge-small)", embedding_speed))
    return 0 if all(results) else 1
