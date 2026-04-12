import torch
import torch.nn as nn
import numpy as np

from src.models.preprocessing import (
    preprocess_email,
    tokens_to_vectors,
    load_word2vec
)

# =========================
# MODEL
# =========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze()


# =========================
# CONFIG
# =========================
device = torch.device("cpu")

MAX_LEN = 20
DIM = 300

WORD2VEC_PATH = "datasets/processed/word2vec.model"
MODEL_PATH = "model.pth"


# =========================
# LOAD MODEL
# =========================
model = LSTMClassifier().to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
threshold = checkpoint["threshold"]

model.eval()

print("✅ Model loaded!")
print("Threshold:", threshold)


# =========================
# LOAD WORD2VEC
# =========================
w2v = load_word2vec(WORD2VEC_PATH)


# =========================
# TEXT → MODEL INPUT
# =========================
def text_to_tensor(text: str):
    tokens = preprocess_email(text)
    vec = tokens_to_vectors(tokens, w2v, MAX_LEN, DIM)

    x = np.array(vec, dtype=np.float32)
    x = x.reshape(1, MAX_LEN, DIM)

    return torch.tensor(x, dtype=torch.float32)


# =========================
# PREDICT
# =========================
def predict(text: str):
    x = text_to_tensor(text).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    pred = 1 if prob >= threshold else 0

    return prob, pred


# =========================
# TEST
# =========================
if __name__ == "__main__":

    text = "Congratulations! You won a free iPhone. Click here now!"

    prob, pred = predict(text)

    print("\n========== RESULT ==========")
    print("Text :", text)
    print("Prob :", prob)
    print("Label:", "SPAM" if pred else "HAM")