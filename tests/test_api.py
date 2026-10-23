import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(bundle_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDRAG_BUNDLE_DIR", str(bundle_dir))
    monkeypatch.setenv("MEDRAG_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MEDRAG_ADMIN_TOKEN", "t0ken")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("MEDRAG_LLM_API_KEY", raising=False)
    from medrag import api, config
    config.get_settings.cache_clear()
    api._RATE.clear()
    with TestClient(api.app) as c:
        yield c
    config.get_settings.cache_clear()


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["generator"] == "extractive" and h["chunks"] == 6


def test_ask_returns_cited_answer_and_logs(client):
    r = client.post("/ask", json={"query": "What causes kidney stones?", "k": 3})
    assert r.status_code == 200
    d = r.json()
    assert d["sources"][0]["focus"] == "Kidney stones" and d["citations"] and d["disclaimer"]
    logs = client.get("/admin/logs", headers={"x-admin-token": "t0ken"}).json()["rows"]
    assert logs[0]["endpoint"] == "ask" and logs[0]["source"] == "user"


def test_search_feedback_metrics(client):
    rid = client.post("/search", json={"query": "glaucoma treatment"}).json()["request_id"]
    assert client.post("/feedback", json={"request_id": rid, "rating": "down"}).status_code == 200
    m = client.get("/metrics").text
    assert 'medrag_requests_total{endpoint="search",status="200"} 1.0' in m
    assert 'medrag_feedback_total{rating="down"} 1.0' in m


def test_validation_auth_and_rate_limit(client):
    assert client.post("/ask", json={"query": "hi"}).status_code == 422
    assert client.get("/admin/logs").status_code == 401
    assert client.post("/admin/flush").status_code == 401
    assert client.post("/admin/flush", headers={"x-admin-token": "t0ken"}).json()["flushed"] == 0  # no bucket
    codes = [client.post("/search", json={"query": "kidney stones"}).status_code for _ in range(22)]
    assert codes.count(429) >= 1
    # admin traffic (the simulator) is exempt and labelled
    r = client.post("/search", json={"query": "kidney stones"},
                    headers={"x-admin-token": "t0ken", "x-traffic-source": "simulated"})
    assert r.status_code == 200
    assert client.get("/admin/logs", headers={"x-admin-token": "t0ken"}).json()["rows"][0]["source"] == "simulated"


def test_home_page(client):
    assert "MedRAG Ops" in client.get("/").text
