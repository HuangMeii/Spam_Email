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

model = None
w2v = None
threshold = 0.5

W_f = None
b_f = None
W_i = None
b_i = None
W_c = None
b_c = None
W_o = None
b_o = None

# =========================
# GLOBAL MODEL
# =========================
model = None
w2v = None
threshold = 0.5


def extract_forget_gate(model):
    W_ih = model.lstm.weight_ih_l0.detach().cpu().numpy()
    b_ih = model.lstm.bias_ih_l0.detach().cpu().numpy()

    hidden_dim = W_ih.shape[0] // 4

    W_f = W_ih[hidden_dim : 2 * hidden_dim]
    b_f = b_ih[hidden_dim : 2 * hidden_dim]

    return W_f, b_f


def load_all():
    global model, w2v, threshold, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o

    print("🔄 Loading model...")

    model = LSTMClassifier().to(device)

    checkpoint = torch.load(str(MODEL_PATH), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = checkpoint.get("threshold", 0.5)

    model.eval()

    print("🔄 Loading Word2Vec...")
    w2v = load_word2vec(str(WORD2VEC_PATH))

    print("🔄 Extracting LSTM forget gate...")

    W_f, b_f = extract_forget_gate(model)
    W_i, b_i = extract_input_gate(model)
    W_c, b_c = extract_candidate_memory(model)
    W_o, b_o = extract_output_gate(model)
    print("✅ Ready!")


def extract_input_gate(model):
    W_ih = model.lstm.weight_ih_l0.detach().cpu().numpy()
    b_ih = model.lstm.bias_ih_l0.detach().cpu().numpy()

    hidden_dim = W_ih.shape[0] // 4

    W_i = W_ih[hidden_dim * 1 : hidden_dim * 2]
    b_i = b_ih[hidden_dim * 1 : hidden_dim * 2]

    return W_i, b_i


def extract_candidate_memory(model):
    W_ih = model.lstm.weight_ih_l0.detach().cpu().numpy()
    b_ih = model.lstm.bias_ih_l0.detach().cpu().numpy()

    hidden_dim = W_ih.shape[0] // 4

    W_c = W_ih[hidden_dim * 2 : hidden_dim * 3]
    b_c = b_ih[hidden_dim * 2 : hidden_dim * 3]

    return W_c, b_c


def extract_output_gate(model):
    W_ih = model.lstm.weight_ih_l0.detach().cpu().numpy()
    b_ih = model.lstm.bias_ih_l0.detach().cpu().numpy()

    hidden_dim = W_ih.shape[0] // 4

    W_o = W_ih[hidden_dim * 3 : hidden_dim * 4]
    b_o = b_ih[hidden_dim * 3 : hidden_dim * 4]

    return W_o, b_o


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
        print("---------------------", W_i, b_i)
        return {
            "raw_text": text,
            "tokens": tokens,
            "vectors": vectors_5d,
            "prob": round(prob, 4),
            "label": "spam" if pred else "ham",
            "W_f": None if W_f is None else W_f.tolist(),
            "b_f": None if b_f is None else b_f.tolist(),
            "W_i": None if W_i is None else W_i.tolist(),
            "b_i": None if b_i is None else b_i.tolist(),
            "W_c": None if W_c is None else W_c.tolist(),
            "b_c": None if b_c is None else b_c.tolist(),
            "W_o": None if W_o is None else W_o.tolist(),
            "b_o": None if b_o is None else b_o.tolist(),
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
