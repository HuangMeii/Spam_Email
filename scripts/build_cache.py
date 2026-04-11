import pandas as pd

from src.data.loaders import build_dataset, load_word2vec, save_cache

DATA_PATH = "/content/drive/MyDrive/email_dataset_github_processed.csv"

W2V_PATH = "/content/drive/MyDrive/embeddings/word2vec.bin"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Loading Word2Vec...")
w2v = load_word2vec(W2V_PATH)

print("Converting dataset...")
X, y = build_dataset(df, w2v)

print("Saving cache...")
save_cache(X, y, "/content/drive/MyDrive/cache/train")

print("✅ Done cache!")
