from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.preprocessing import (
    load_word2vec,
    preprocess_email,
    tokens_to_vectors,
)


# =========================
# MODEL
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze()


# =========================
# PATH
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

WORD2VEC_PATH = BASE_DIR / "datasets" / "processed" / "word2vec.model"
MODEL_PATH = BASE_DIR / "model.pth"

device = torch.device("cpu")

MAX_LEN = 20
DIM = 300


# =========================
# GLOBAL MODEL
# =========================
model = None
w2v = None
threshold = 0.5


def load_all():
    global model, w2v, threshold

    print("🔄 Loading model...")

    model = LSTMClassifier().to(device)

    checkpoint = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = checkpoint.get("threshold", 0.5)

    model.eval()

    print("✅ Model loaded!")

    print("🔄 Loading Word2Vec...")
    w2v = load_word2vec(str(WORD2VEC_PATH))
    print("✅ Word2Vec loaded!")


# =========================
# MAIN PREDICT FUNCTION
# =========================
def predict_text(text: str):
    try:
        print("👉 Input:", text)

        if model is None or w2v is None:
            load_all()

        # =========================
        # 1. PREPROCESS
        # =========================
        tokens = preprocess_email(text)

        # =========================
        # 2. WORD2VEC
        # =========================
        vectors = tokens_to_vectors(tokens, w2v, MAX_LEN, DIM)

        vectors_5d = [vec[:5].tolist() for vec in vectors[: len(tokens)]]

        # =========================
        # 3. MODEL INPUT
        # =========================
        x = np.array(vectors, dtype=np.float32)
        x = x.reshape(1, MAX_LEN, DIM)
        x = torch.tensor(x, dtype=torch.float32).to(device)

        # =========================
        # 4. PREDICT
        # =========================
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()

        pred = 1 if prob >= threshold else 0

        # =========================
        # RETURN FULL PIPELINE
        # =========================
        return {
            "raw_text": text,
            "tokens": tokens,
            "vectors": vectors_5d,
            "prob": round(prob, 4),
            "label": "spam" if pred else "ham",
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# TEST LOCAL
# =========================
if __name__ == "__main__":
    load_all()

    text = input("Enter email: ")
    result = predict_text(text)

    print("\n========== RESULT ==========")
    print(result)
