from pathlib import Path

import numpy as np
import pandas as pd


def load_vector_dataset(
    dataset_path: str | Path, vector_prefix: str = "w2v_", label_col: str = "isSpam"
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(dataset_path)

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    vector_cols = [c for c in df.columns if c.startswith(vector_prefix)]

    if not vector_cols:
        raise ValueError("No vector columns found!")

    X = df[vector_cols].values
    y = df[label_col].values

    return X, y
