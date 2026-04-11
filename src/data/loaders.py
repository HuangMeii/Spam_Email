from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize


def load_word2vec(path=None):
    if path is None:
        path = "/content/drive/MyDrive/embeddings/word2vec.bin"


def text_to_vector(
    text: str,
    w2v: KeyedVectors,
    max_len: int,
) -> np.ndarray:
    """
    Convert text to sequence of word vectors.
    """
    tokens = word_tokenize(str(text).lower())

    vectors = [w2v[t] for t in tokens if t in w2v]

    embed_dim = w2v.vector_size

    if len(vectors) < max_len:
        vectors += [np.zeros(embed_dim)] * (max_len - len(vectors))
    else:
        vectors = vectors[:max_len]

    return np.array(vectors, dtype=np.float32)


def build_dataset(
    df: pd.DataFrame,
    w2v: KeyedVectors,
    text_col: str = "msg",
    label_col: str = "isSpam",
    max_len: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert dataframe to (X, y).
    """
    X = np.array(
        [text_to_vector(t, w2v, max_len) for t in df[text_col]],
        dtype=np.float32,
    )

    y = df[label_col].values.astype(np.float32)

    return X, y


def save_cache(X: np.ndarray, y: np.ndarray, prefix: str) -> None:
    """
    Save dataset to disk.
    """
    np.save(f"{prefix}_X.npy", X)
    np.save(f"{prefix}_y.npy", y)


def load_cache(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from cache.
    """
    X = np.load(f"{prefix}_X.npy")
    y = np.load(f"{prefix}_y.npy")
    return X, y
