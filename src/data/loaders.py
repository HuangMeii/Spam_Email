from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


# =========================
# WORD2VEC LOADER
# =========================
def load_word2vec(path: str) -> KeyedVectors:
    """
    Load pre-trained Word2Vec model.
    """
    return KeyedVectors.load_word2vec_format(
        path,
        binary=True,
        limit=100000,  # 🔥 speed + RAM optimization
    )


# =========================
# TOKENIZER (FAST + NO NLTK)
# =========================
def tokenize(text: str) -> list[str]:
    """
    Fast tokenizer (replace nltk.word_tokenize for performance).
    """
    return str(text).lower().split()


# =========================
# TEXT → VECTOR
# =========================
def text_to_vector(text: str, w2v: KeyedVectors, max_len: int) -> np.ndarray:
    """
    Convert text into (max_len, embedding_dim) matrix.
    """

    tokens = tokenize(text)

    embed_dim = w2v.vector_size
    vocab = w2v.key_to_index  # 🔥 O(1) lookup

    vectors: list[np.ndarray] = []

    for t in tokens[:max_len]:
        if t in vocab:
            vectors.append(w2v[t])

    # handle empty text
    if len(vectors) == 0:
        return np.zeros((max_len, embed_dim), dtype=np.float32)

    # padding
    if len(vectors) < max_len:
        pad = np.zeros(embed_dim, dtype=np.float32)
        vectors += [pad] * (max_len - len(vectors))
    else:
        vectors = vectors[:max_len]

    return np.array(vectors, dtype=np.float32)


# =========================
# BUILD DATASET
# =========================
def build_dataset(
    df: pd.DataFrame,
    w2v: KeyedVectors,
    text_col: str = "msg",
    label_col: str = "isSpam",
    max_len: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert dataframe → (X, y)
    """

    texts = df[text_col].values
    labels = df[label_col].values.astype(np.float32)

    embed_dim = w2v.vector_size

    X = np.zeros((len(texts), max_len, embed_dim), dtype=np.float32)

    vocab = w2v.key_to_index

    for i, text in enumerate(texts):
        tokens = tokenize(text)

        vecs: list[np.ndarray] = []

        for t in tokens[:max_len]:
            if t in vocab:
                vecs.append(w2v[t])

        if len(vecs) == 0:
            continue

        # pad
        if len(vecs) < max_len:
            pad = np.zeros(embed_dim, dtype=np.float32)
            vecs += [pad] * (max_len - len(vecs))
        else:
            vecs = vecs[:max_len]

        X[i] = np.array(vecs, dtype=np.float32)

    return X, labels


# =========================
# SAVE CACHE
# =========================
def save_cache(X: np.ndarray, y: np.ndarray, prefix: str) -> None:
    """
    Save numpy dataset to disk.
    """
    np.save(f"{prefix}_X.npy", X)
    np.save(f"{prefix}_y.npy", y)


# =========================
# LOAD CACHE
# =========================
def load_cache(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load cached dataset.
    """
    X = np.load(f"{prefix}_X.npy")
    y = np.load(f"{prefix}_y.npy")
    return X, y
