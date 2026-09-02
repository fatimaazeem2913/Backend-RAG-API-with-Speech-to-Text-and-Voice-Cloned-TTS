import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_get_sources_endpoint():
    response = client.get("/api/rag/sources")
    assert response.status_code == 200
    assert "sources" in response.json()
    assert isinstance(response.json()["sources"], list)


def test_chat_endpoint_validation():
    # Empty message validation
    response = client.post(
        "/api/rag/chat",
        json={"session_id": "test_session", "message": "   "}
    )
    assert response.status_code == 400


def test_multi_turn_chat_and_citations():
    session_id = "test_multi_turn"

    # Turn 1
    t1_payload = {
        "session_id": session_id,
        "message": "What is the formula for Mean Squared Error?",
        "strategy": "hybrid"
    }
    t1_res = client.post("/api/rag/chat", json=t1_payload)
    assert t1_res.status_code == 200
    data1 = t1_res.json()
    assert "answer" in data1
    assert "citations" in data1
    assert isinstance(data1["citations"], list)

    # Turn 2
    t2_payload = {
        "session_id": session_id,
        "message": "What do the variables in the summation represent?",
        "strategy": "hybrid"
    }
    t2_res = client.post("/api/rag/chat", json=t2_payload)
    assert t2_res.status_code == 200
    data2 = t2_res.json()
    assert "standalone_query" in data2

    # Reset Session
    reset_res = client.post("/api/rag/session/reset", json={"session_id": session_id})
    assert reset_res.status_code == 200