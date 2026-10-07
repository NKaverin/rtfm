import os
from pathlib import Path

import numpy as np
import wordllama
from google import genai
from google.genai import types

BACKEND = os.environ.get("EMBEDDING_BACKEND", "local")
DIMS = {"local": 256, "gemini":768}
if BACKEND not in DIMS:
    raise ValueError(f"EMBEDDING_BACKEND must be one of {list(DIMS)}")
DIM = DIMS[BACKEND]

GEMINI_MODEL = "gemini-embedding-001"
GEMINI_BATCH_SIZE = 100

_local_model = wordllama.WordLlama.load(
    cache_dir=Path(wordllama.__file__).parent, disable_download=True
)

def _embed_local(texts: list[str]) -> np.ndarray:
    return _local_model.embed(texts)

def _embed_gemini(texts: list[str]) -> np.ndarray:
    client = genai.Client(
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=5))
    )
    vectors = []
    for start in range(0, len(texts), GEMINI_BATCH_SIZE):
        batch = texts[start : start + GEMINI_BATCH_SIZE]
        result = client.models.embed_content(
            model=GEMINI_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(output_dimensionality=DIMS["gemini"])
        )
        vectors.extend(e.values for e in result.embeddings)
    return np.array(vectors, dtype=np.float32)

def embed(texts: list[str]) -> np.ndarray:
    if BACKEND == "gemini":
        return _embed_gemini(texts)
    return _embed_local(texts)

def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lengths = np.linalg.norm(a) * np.linalg.norm(b)
    if lengths == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(a @ b / lengths)

def count_tokens(text: str) -> int:
    return len(_local_model.tokenize(text)[0].ids)

if __name__ == "__main__":
    base = "How do I undo my last commit?"
    others = [
        "How do I undo my last commit?",
        "How can I undo the last commit?",
        "how do i undo my last commit",
        "What's the way to undo my most recent commit?",
        "I want to revert the commit I just made, how?",
        "How do I see my last commit?",
        "How do I undo my last push?",
        "How do I redo my last commit?",
        "How do I undo my last merge?",
        "How do I create a branch?",
    ]
    vectors = embed([base] + others)
    for text, vector in zip(others, vectors[1:]):
        print(f"{1 - cosine_similarity(vectors[0], vector):.3f}  {text}")