from __future__ import annotations

import os
import re
from collections.abc import Iterable
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer

# =========================
# STOPWORDS
# =========================
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "while",
        "with",
        "to",
        "of",
        "at",
        "by",
        "for",
        "from",
        "in",
        "on",
        "off",
        "out",
        "over",
        "under",
        "as",
        "is",
        "it",
        "this",
        "that",
        "these",
        "those",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "so",
        "such",
        "no",
        "not",
        "too",
        "very",
        "can",
        "will",
    }
)

_STEMMER = PorterStemmer()

_HTML_PATTERN = re.compile(r"<[^>]+>")
_NON_WORD_PATTERN = re.compile(r"[^\w\s]")


# =========================
# LOAD DATA
# =========================
def load_email_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, encoding="latin1")
    df["msg"] = df["msg"].fillna("").astype(str)
    df["isSpam"] = df["isSpam"].astype(int)
    return df


# =========================
# CLEAN + TOKENIZE
# =========================
def preprocess_email(text: str) -> list[str]:
    text = unescape(text.lower())
    text = _HTML_PATTERN.sub(" ", text)
    text = _NON_WORD_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for t in text.split():
        if t not in STOPWORDS:
            tokens.append(_STEMMER.stem(t))

    return tokens


def tokenize_texts(texts: Iterable[str]) -> list[list[str]]:
    return [preprocess_email(t) for t in texts]


# =========================
# WORD2VEC TRAIN
# =========================
def train_word2vec(tokenized: list[list[str]], vector_size=300) -> Word2Vec:
    workers = max(1, (os.cpu_count() or 2) - 1)

    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=5,
        min_count=1,
        sg=1,
        workers=workers,
        epochs=5,
        seed=42,
    )

    return model


# =========================
# LOAD WORD2VEC (FOR PREDICT)
# =========================
def load_word2vec(path: str) -> Word2Vec:
    return Word2Vec.load(path)


# =========================
# TOKENS → VECTOR SEQUENCE
# =========================
def tokens_to_vectors(tokens, model, max_len=20, dim=300):
    seq = []

    for t in tokens[:max_len]:
        if t in model.wv:
            seq.append(model.wv[t])
        else:
            seq.append(np.zeros(dim, dtype=np.float32))

    # padding
    while len(seq) < max_len:
        seq.append(np.zeros(dim, dtype=np.float32))

    return seq


# =========================
# BUILD DATASET (TRAIN)
# =========================
def build_dataset(input_path: str, output_dir: str, max_len: int = 20):
    print("Loading dataset...")
    df = load_email_dataframe(input_path)

    print("Tokenizing...")
    tokenized = tokenize_texts(df["msg"])

    print("Training Word2Vec...")
    w2v = train_word2vec(tokenized)
    dim = w2v.vector_size

    print("Converting to sequences...")

    X = np.array(
        [tokens_to_vectors(tokens, w2v, max_len, dim) for tokens in tokenized],
        dtype=np.float32,
    )

    y = df["isSpam"].values.astype(np.int64)

    # =========================
    # SAVE
    # =========================
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    w2v.save(str(output_dir / "word2vec.model"))

    print("\nDONE ✅")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Saved to:", output_dir)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    build_dataset(
        input_path="datasets/processed/email_dataset_github_processed.csv",
        output_dir="datasets/processed/",
        max_len=20,
    )
