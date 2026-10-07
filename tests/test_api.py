import redis
from fastapi.testclient import TestClient

import answer
import ingest
import memory
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

    def fake_answer(question, session_id, user_id):
        questions.append((question, session_id, user_id))
        return {
            "answer": "Use git revert [undoing.asc].",
            "sources": ["undoing.asc"],
            "refused": False,
            "cached": False,
            "remembered": [],
        }

    monkeypatch.setattr(answer, "answer", fake_answer)
    body = {"question": "  How do I undo a commit?  ", "session_id": "abc-123", "user_id": "nik"}
    response = client.post("/ask", json=body)

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Use git revert [undoing.asc].",
        "sources": ["undoing.asc"],
        "refused": False,
        "cached": False,
        "remembered": [],
        "session_id": "abc-123",
    }
    assert questions == [("How do I undo a commit?", "abc-123", "nik")]

def test_ask_rejects_a_blank_question_without_calling_the_model(monkeypatch):
    def must_not_run(question, session_id, user_id):
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

def test_ask_creates_a_session_id_when_none_is_sent(monkeypatch):
    seen = []

    def fake_answer(question, session_id, user_id):
        seen.append(session_id)
        return {"answer": "x", "sources": [], "refused": False, "cached": False, "remembered": []}
    
    monkeypatch.setattr(answer, "answer", fake_answer)
    response = client.post("/ask", json={"question": "How do I undo a commit?"})

    session_id = response.json()["session_id"]
    assert len(session_id) == 32
    assert seen == [session_id]


def test_ask_rejects_a_session_id_with_unsafe_characters():
    response = client.post("/ask", json={"question": "x", "session_id": "a b:c"})
    assert response.status_code == 422

def test_memories_can_be_listed_and_deleted(isolated_memory):
    memory.remember("nik", [{"text": "The user writes Go.", "topics": ["language"], "entities": ["Go"]}])

    listed = client.get("/memories/nik")
    assert listed.json() == [{"text": "The user writes Go.", "topics": ["language"], "entities": ["Go"]}]

    assert client.delete("/memories/nik").json() == {"removed": 1}
    assert client.get("/memories/nik").json() == []


def test_memories_rejects_a_user_id_with_unsafe_characters():
    assert client.get("/memories/a-b").status_code == 422    