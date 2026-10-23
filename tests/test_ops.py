import json
from datetime import datetime, timezone

import boto3
import pytest
from moto.server import ThreadedMotoServer

from medrag.feedback import parse_grades
from medrag.logsink import RequestLogger, load_logs
from medrag.monitor import compute_report
from medrag.registry import _decide

PARAMS = {
    "monitoring": {"window_days": 7, "min_queries": 30, "drift_auc_threshold": 0.75,
                   "max_low_confidence_rate": 0.25, "max_p95_latency_ms": 1500, "max_error_rate": 0.02},
    "promotion": {"metric": "liveqa_test/ndcg@10", "min_improvement": 0.0,
                  "guardrails": {"medquad/recall@5": 0.7}},
}


def _log(q, score=2.0, low=False):
    return {"ts": datetime.now(timezone.utc).isoformat(), "endpoint": "ask", "query": q, "top_score": score,
            "low_confidence": low, "latency_ms": 12.0, "gen_mode": "extractive", "source": "user"}


REF = [{"query": q, "top_score": 2.0} for q in
       [f"What are the symptoms of {d} ?" for d in ("Gaucher disease", "glaucoma", "asthma", "gout", "lupus")] * 12 +
       [f"What are the treatments for {d} ?" for d in ("Gaucher disease", "glaucoma", "asthma", "gout", "lupus")] * 12]


def test_monitor_insufficient_then_healthy_then_drift():
    rep = compute_report(PARAMS, [_log("What are the symptoms of gout ?")] * 5, REF, {})
    assert rep["status"] == "insufficient_data" and not rep["retrain_recommended"]

    same = [_log(r["query"]) for r in REF[:60]]
    rep = compute_report(PARAMS, same, REF, {})
    assert rep["status"] == "healthy", rep["reasons"]

    drifted = [_log(f"hi doc my {x} has been taking pills for months is it safe to stop thank you") for x in
               ("mom", "dad", "son", "wife", "friend", "aunt")] * 10
    rep = compute_report(PARAMS, drifted, REF, {})
    assert rep["retrain_recommended"] and rep["drift"]["auc"] > 0.75
    assert any(t in dict(rep["drift"]["production_terms"]) for t in ("taking", "pills", "thank"))


def test_promotion_rules():
    good = {"medquad/recall@5": 0.8, "liveqa_test/ndcg@10": 0.45}
    assert _decide(PARAMS, good, None)[0]                                        # first model
    assert _decide(PARAMS, good, {**good, "liveqa_test/ndcg@10": 0.40})[0]      # strictly better
    assert not _decide(PARAMS, good, dict(good))[0]                             # tie: keep champion
    assert _decide(PARAMS, good, dict(good), data_changed=True)[0]              # tie on new corpus: refresh
    assert not _decide(PARAMS, {**good, "medquad/recall@5": 0.5}, None)[0]      # guardrail


def test_parse_grades():
    assert parse_grades('sure: {"grades": [3, 0, 5]}', 3) == [3, 0, 3]
    assert parse_grades("not json", 2) is None
    assert parse_grades('{"grades": [1]}', 2) is None


@pytest.fixture
def s3_endpoint():
    """A local S3-compatible server (like DagsHub storage or R2 in production)."""
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


def test_logs_roundtrip_through_s3(tmp_path, s3_endpoint):
    boto3.client("s3", endpoint_url=s3_endpoint, region_name="us-east-1").create_bucket(Bucket="logs")
    lg = RequestLogger(tmp_path, bucket="logs", flush_records=2, flush_seconds=999, endpoint_url=s3_endpoint)
    for i in range(3):
        lg.log(_log(f"question {i}"))
    lg.close()
    assert lg.uploaded_batches == 2 and lg.upload_errors == 0
    rows = load_logs(1, bucket="logs", endpoint_url=s3_endpoint)
    assert sorted(r["query"] for r in rows) == ["question 0", "question 1", "question 2"]
    assert len((tmp_path / f"requests-{datetime.now(timezone.utc):%Y%m%d}.jsonl").read_text().splitlines()) == 3


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch, tmp_path):
    # isolate from any real AWS config/credentials on the machine running the tests
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws-creds"))
    monkeypatch.delenv("AWS_CA_BUNDLE", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def test_report_json_serialisable():
    rep = compute_report(PARAMS, [_log(r["query"]) for r in REF[:40]], REF, {"model_version": "m/v1"})
    json.dumps(rep)
