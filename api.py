from typing import Annotated

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StringConstraints

import answer
import ingest
import store

app = FastAPI(title="RTFM")


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    refused: bool

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