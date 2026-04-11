import pandas as pd

INPUT = "/content/drive/MyDrive/email_dataset_github_processed.csv"

OUTPUT = "/content/drive/MyDrive/datasets/processed.csv"

df = pd.read_csv(INPUT)

# ví dụ clean cơ bản
df["msg"] = df["msg"].astype(str).str.lower().str.strip()

df.to_csv(OUTPUT, index=False)

print("✅ Saved processed dataset")
