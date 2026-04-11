# scripts/converter.py

import pandas as pd

from src.data.loaders import build_dataset, load_word2vec, save_cache
from src.data.splitter import stratified_train_val_test_split


def main() -> None:
    # 1. Load dataset
    df = pd.read_csv("/content/drive/MyDrive/email_dataset_github_processed.csv")

    # 2. Split dataset
    train_df, val_df, test_df = stratified_train_val_test_split(df)

    # 3. Load Word2Vec
    w2v = load_word2vec("word2vec.bin")

    # 4. Convert → vector
    print("Converting train set...")
    X_train, y_train = build_dataset(train_df, w2v)

    print("Converting val set...")
    X_val, y_val = build_dataset(val_df, w2v)

    print("Converting test set...")
    X_test, y_test = build_dataset(test_df, w2v)

    # 5. Save cache
    save_cache(X_train, y_train, "cache/train")
    save_cache(X_val, y_val, "cache/val")
    save_cache(X_test, y_test, "cache/test")

    print("Done ✅ Dataset cached.")


if __name__ == "__main__":
    main()
