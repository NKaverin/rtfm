import json
from collections.abc import Iterator
from typing import Annotated

import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, StringConstraints

import answer
import cache
import ingest
import store

app = FastAPI(title="RTFM")


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    refused: bool
    cached: bool

@app.get("/metrics")
def metrics() -> dict:
    return cache.metrics()


@app.post("/cache/flush")
def flush_cache() -> dict:
    return {"removed": cache.flush()}
            
def _sse(events: Iterator[dict]) -> Iterator[str]:
    try:
        for event in events:
            data = {key: value for key, value in event.items() if key != "type"}
            yield f"event: {event['type']}\ndata: {json.dumps(data)}\n\n"
    except Exception as error:
        data = {"message": f"{type(error).__name__}: {error}"}
        yield f"event: error\ndata: {json.dumps(data)}\n\n"

@app.get("/health")
def health() -> dict:
    try:
        store.r.ping()
    except redis.ConnectionError:
        raise HTTPException(status_code=503, detail="Redis is not reachable")
    return {"status": "ok"}

@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    return answer.answer(request.question)

@app.post("/ingest")
def run_ingest() -> dict:
    return ingest.ingest_book(ingest.BOOK_DIR)

@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    hits = answer.retrieve(request.question)
    events = answer.answer_events(request.question, hits)
    return StreamingResponse(_sse(events), media_type="text/event-stream")