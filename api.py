import json
from collections.abc import Iterator
from typing import Annotated

import redis
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

import answer
import cache
import ingest
import memory
import sessions
import store

app = FastAPI(title="RTFM")


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    user_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_]{1,64}$")
    source: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_./-]{2,200}$")

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    refused: bool
    cached: bool
    remembered: list[str]    
    session_id: str

@app.get("/metrics")
def metrics() -> dict:
    return cache.metrics()


@app.post("/cache/flush")
def flush_cache() -> dict:
    return {"removed": cache.flush()}
USER_ID = Annotated[str, Path(pattern=r"^[A-Za-z0-9_]{1,64}$")]


@app.get("/memories/{user_id}")
def list_memories(user_id: USER_ID) -> list[dict]:
    return memory.all_memories(user_id)


@app.delete("/memories/{user_id}")
def delete_memories(user_id: USER_ID) -> dict:
    return {"removed": memory.forget(user_id)}
    
def _sse(events: Iterator[dict]) -> Iterator[str]:
    try:
        for event in events:
            data = {key: value for key, value in event.items() if key != "type"}
            yield f"event: {event['type']}\ndata: {json.dumps(data)}\n\n"
    except Exception as error:
        data = {"message": f"{type(error).__name__}: {error}"}
        yield f"event: error\ndata: {json.dumps(data)}\n\n"

@app.exception_handler(answer.ModelError)
def model_failed(request, error) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": f"The model call failed. {error}"})


@app.exception_handler(redis.ConnectionError)
def redis_failed(request, error) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Redis is not reachable"})

@app.get("/health")
def health() -> dict:
    try:
        store.r.ping()
    except redis.ConnectionError:
        raise HTTPException(status_code=503, detail="Redis is not reachable")
    return {"status": "ok"}

@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    session_id = request.session_id or sessions.new_id()
    result = answer.answer(request.question, session_id, request.user_id, request.source)
    return {**result, "session_id": session_id}

@app.get("/sources")
def list_sources() -> list[str]:
    return store.sources()

@app.post("/ingest")
def run_ingest() -> dict:
    return ingest.ingest_book(ingest.BOOK_DIR)

@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    hits = answer.retrieve(request.question)
    events = answer.answer_events(request.question, hits)
    return StreamingResponse(_sse(events), media_type="text/event-stream")