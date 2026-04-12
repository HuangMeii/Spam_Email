# import numpy as np
# import torch
# import torch.nn as nn

# from src.models.preprocessing import load_word2vec, preprocess_email, tokens_to_vectors


# # =========================
# # MODEL
# # =========================
# class LSTMClassifier(nn.Module):
#     def __init__(self, input_dim=300, hidden_dim=64):
#         super().__init__()

#         self.lstm = nn.LSTM(
#             input_size=input_dim, hidden_size=hidden_dim, batch_first=True
#         )

#         self.fc = nn.Linear(hidden_dim, 1)

#     def forward(self, x):
#         _, (h_n, _) = self.lstm(x)
#         return self.fc(h_n[-1]).squeeze()


# # =========================
# # CONFIG
# # =========================
# device = torch.device("cpu")

# MAX_LEN = 20
# DIM = 300

# WORD2VEC_PATH = "datasets/processed/word2vec.model"
# MODEL_PATH = "model.pth"


# # =========================
# # LOAD MODEL
# # =========================
# model = LSTMClassifier().to(device)

# checkpoint = torch.load(MODEL_PATH, map_location=device)

# model.load_state_dict(checkpoint["model_state_dict"])
# threshold = checkpoint["threshold"]

# model.eval()

# print("✅ Model loaded!")
# print("Threshold:", threshold+0.1)


# # =========================
# # LOAD WORD2VEC
# # =========================
# w2v = load_word2vec(WORD2VEC_PATH)


# # =========================
# # TEXT → MODEL INPUT
# # =========================
# def text_to_tensor(text: str):
#     tokens = preprocess_email(text)
#     vec = tokens_to_vectors(tokens, w2v, MAX_LEN, DIM)

#     x = np.array(vec, dtype=np.float32)
#     x = x.reshape(1, MAX_LEN, DIM)

#     return torch.tensor(x, dtype=torch.float32)


# # =========================
# # PREDICT
# # =========================
# def predict(text: str):
#     x = text_to_tensor(text).to(device)

#     with torch.no_grad():
#         logits = model(x)
#         prob = torch.sigmoid(logits).item()

#     pred = 1 if prob >= threshold else 0

#     return prob, pred


# # =========================
# # TEST
# # =========================
# if __name__ == "__main__":
#     text=input("Testing the model with a sample email...")

#     prob, pred = predict(text)

#     print("\n========== RESULT ==========")
#     print("Text :", text)
#     print("Prob :", prob)
#     print("Label:", "SPAM" if pred else "HAM")
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

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
# PATH (FIX CHUẨN)
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

WORD2VEC_PATH = BASE_DIR / "datasets" / "processed" / "word2vec.model"
MODEL_PATH = BASE_DIR / "model.pth"

device = torch.device("cpu")

MAX_LEN = 20
DIM = 300


# =========================
# GLOBAL (load 1 lần)
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
# TEXT → TENSOR
# =========================
def text_to_tensor(text: str):
    tokens = preprocess_email(text)
    vec = tokens_to_vectors(tokens, w2v, MAX_LEN, DIM)

    x = np.array(vec, dtype=np.float32)
    x = x.reshape(1, MAX_LEN, DIM)

    return torch.tensor(x, dtype=torch.float32)


# =========================
# PREDICT FUNCTION
# =========================
def predict_text(text: str):
    try:
        print("👉 Input:", text)

        if model is None or w2v is None:
            print("⚠️ Loading model...")
            load_all()

        tokens = preprocess_email(text)
        print("Tokens:", tokens)

        x = text_to_tensor(text).to(device)

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).item()

        pred = 1 if prob >= threshold else 0

        result = {
            "prob": round(prob, 4),
            "label": "spam" if pred else "ham"
        }

        print("✅ Result:", result)

        return result

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}

# =========================
# TEST LOCAL
# =========================
if __name__ == "__main__":
    load_all()

    text = input("Enter email: ")
    result = predict_text(text)

    print("\n========== RESULT ==========")
    print("Text :", text)
    print("Prob :", result["prob"])
    print("Label:", result["label"])