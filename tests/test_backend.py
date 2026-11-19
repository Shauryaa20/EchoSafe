from fastapi.testclient import TestClient

from backend.main import app, engine


client = TestClient(app)


def test_health_reports_demo_fallback_without_checkpoint(monkeypatch):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] in {"checkpoint", "demo_fallback"}


def test_analyze_is_deterministic_and_labelled():
    first = client.post("/api/analyze", json={"text": "I feel anxious and worried."})
    second = client.post("/api/analyze", json={"text": "I feel anxious and worried."})
    assert first.status_code == 200
    body = first.json()
    assert body == second.json()
    assert body["mode"] == "checkpoint"
    assert body["model_status"] == "available"
    assert len(body["emotions"]) == 7
    assert 0.0 <= body["stress_score"] <= 1.0


def test_checkpoint_is_loaded_and_distinct_inputs_reach_model():
    assert engine.status == "available"
    calm = client.post("/api/analyze", json={"text": "I feel calm and grateful today."}).json()
    angry = client.post("/api/analyze", json={"text": "I am furious and overwhelmed by this situation."}).json()
    assert calm["mode"] == angry["mode"] == "checkpoint"
    assert calm["emotions"] != angry["emotions"] or calm["stress_score"] != angry["stress_score"]


def test_analyze_conversation():
    response = client.post("/api/analyze/conversation", json={
        "messages": [
            {"role": "user", "text": "Today was difficult."},
            {"role": "assistant", "text": "What made it difficult?"},
            {"role": "user", "text": "I feel scared."},
        ]
    })
    assert response.status_code == 200
    assert response.json()["analysis_type"] == "contextual"
    assert response.json()["context"] == ["Today was difficult.", "What made it difficult?"]


def test_experiments_and_resources():
    assert client.get("/api/experiments").status_code == 200
    assert client.get("/api/experiments/plots").status_code == 200
    assert client.get("/api/resources").status_code == 200


def test_history_round_trip():
    created = client.post("/api/history", json={
        "text": "An opt-in test entry",
        "analysis_type": "contextual",
        "context": ["Earlier context"],
        "emotions": [{"label": "neutral", "score": 0.8}],
        "stress_score": 0.2,
    })
    assert created.status_code == 200
    entry = created.json()
    assert entry["id"]
    listed = client.get("/api/history")
    assert any(row["id"] == entry["id"] for row in listed.json()["entries"])
    saved = next(row for row in listed.json()["entries"] if row["id"] == entry["id"])
    assert saved["analysis_type"] == "contextual"
    assert saved["context"] == ["Earlier context"]
    deleted = client.delete(f"/api/history/{entry['id']}")
    assert deleted.status_code == 200


def test_app_main_import_compatibility():
    from backend.app.main import app as compatible_app

    assert compatible_app is app
