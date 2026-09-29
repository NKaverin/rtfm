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

_local_model = wordllama.WordLlama.load(
    cache_dir=Path(wordllama.__file__).parent, disable_download=True
)

def _embed_local(texts: list[str]) -> np.ndarray:
    return _local_model.embed(texts)

def _embed_gemini(texts: list[str]) -> np.ndarray:
    client = genai.Client()
    result = client.models.embed_content(
        model=GEMINI_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=DIMS["gemini"])
    )
    return np.array([e.values for e in result.embeddings], dtype=np.float32)

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

if __name__ == "__main__":
    question = "How do I undo my last commit?"
    sentences = [
        "git reset HEAD~1 moves the branch back one commit",
        "git commit --amend replaces the last commit with a new one",
        "git revert creates a new commit that undoes an earlier one",
        "git branch lists, creates, or deletes branches",
        "git clone copies a remote repository to your machine",
        "Bananas are rich in potassium",
    ]
    print("backend:", BACKEND, "| dim:", DIM)
    q_vec = embed([question])[0]
    s_vecs = embed(sentences)
    print("shape:", s_vecs.shape, s_vecs.dtype)
    scored = [(cosine_similarity(q_vec, v), text) for v, text in zip(s_vecs, sentences)]
    for score, text in sorted(scored, reverse=True):
        print(f"{score:.3f}  {text}")