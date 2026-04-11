import pandas as pd

from src.data.loaders import build_dataset, load_word2vec, save_cache
from src.data.splitter import stratified_train_val_test_split

# load data
df = pd.read_csv("/content/drive/MyDrive/email_dataset_github_processed.csv")

# split
train_df, val_df, test_df = stratified_train_val_test_split(df)

# load w2v
w2v = load_word2vec("word2vec.bin")

# convert + cache
X_train, y_train = build_dataset(train_df, w2v)
X_val, y_val = build_dataset(val_df, w2v)

save_cache(X_train, y_train, "cache/train")
save_cache(X_val, y_val, "cache/val")
