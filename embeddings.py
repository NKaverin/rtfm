from pathlib import Path

import numpy as np
import wordllama
from wordllama import WordLlama

DIM = 256

_model = WordLlama.load(cache_dir=Path(wordllama.__file__).parent, disable_download=True)

def embed(texts: list[str]) -> np.ndarray:
    return _model.embed(texts)

def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lengths = np.linalg.norm(a) * np.linalg.norm(b)
    if lengths == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(a @ b / lengths)