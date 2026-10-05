import redis
from fastapi.testclient import TestClient

import answer
import ingest
import store
from api import app

client = TestClient(app)


def test_health_reports_ok_when_redis_answers():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_reports_503_when_redis_is_down(monkeypatch):
    def refuse():
        raise redis.ConnectionError("down")

    monkeypatch.setattr(store.r, "ping", refuse)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"detail": "Redis is not reachable"}


def test_ask_returns_the_answer_as_json(monkeypatch):
    questions = []

    def fake_answer(question):
        questions.append(question)
        return {"answer": "Use git revert [undoing.asc].", "sources": ["undoing.asc"], "refused": False}

    monkeypatch.setattr(answer, "answer", fake_answer)
    response = client.post("/ask", json={"question": "  How do I undo a commit?  "})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Use git revert [undoing.asc].",
        "sources": ["undoing.asc"],
        "refused": False,
    }
    assert questions == ["How do I undo a commit?"]


def test_ask_rejects_a_blank_question_without_calling_the_model(monkeypatch):
    def must_not_run(question):
        raise AssertionError("answer() was called")

    monkeypatch.setattr(answer, "answer", must_not_run)
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_rejects_a_request_without_a_question():
    assert client.post("/ask", json={"q": "x"}).status_code == 422


def test_ingest_runs_the_ingest_and_returns_its_counts(monkeypatch):
    monkeypatch.setattr(ingest, "ingest_book", lambda book_dir: {"chunks": 5, "removed": 1})
    response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json() == {"chunks": 5, "removed": 1}

def test_ask_stream_sends_each_event_in_sse_format(monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda question: ["a hit"])

    def fake_events(question, hits):
        assert hits == ["a hit"]
        yield {"type": "token", "text": "Use git revert."}
        yield {"type": "done", "sources": ["undoing.asc"], "refused": False}

    monkeypatch.setattr(answer, "answer_events", fake_events)
    response = client.post("/ask/stream", json={"question": "How do I undo a commit?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == (
        'event: token\ndata: {"text": "Use git revert."}\n\n'
        'event: done\ndata: {"sources": ["undoing.asc"], "refused": false}\n\n'
    )


def test_ask_stream_reports_a_failure_mid_answer_as_an_error_event(monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda question: [])

    def failing_events(question, hits):
        yield {"type": "token", "text": "Use "}
        raise RuntimeError("model went away")

    monkeypatch.setattr(answer, "answer_events", failing_events)
    response = client.post("/ask/stream", json={"question": "How do I undo a commit?"})

    assert response.status_code == 200
    assert response.text.endswith('event: error\ndata: {"message": "RuntimeError: model went away"}\n\n')


def test_ask_stream_fails_before_streaming_when_retrieval_fails(monkeypatch):
    def broken_retrieve(question):
        raise redis.ConnectionError("down")

    monkeypatch.setattr(answer, "retrieve", broken_retrieve)
    quiet_client = TestClient(app, raise_server_exceptions=False)
    response = quiet_client.post("/ask/stream", json={"question": "How do I undo a commit?"})
    assert response.status_code == 500    