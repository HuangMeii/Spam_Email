from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.models.predict import predict_text, text_to_tensor
from src.models.preprocessing import preprocess_email, tokens_to_vectors, load_word2vec

import numpy as np

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD WORD2VEC (1 lần)
# =========================
WORD2VEC_PATH = "datasets/processed/word2vec.model"
w2v = load_word2vec(WORD2VEC_PATH)

MAX_LEN = 20
DIM = 300

# =========================
# SCHEMA
# =========================
class InputText(BaseModel):
    text: str

# =========================
# ROOT
# =========================
@app.get("/")
def home():
    return {"msg": "API is running 🚀"}

# =========================
# PREDICT + PIPELINE
# =========================
@app.post("/predict")
def predict_api(data: InputText):
    text = data.text

    print("📩 Input:", text)

    # ===== 1. PREPROCESS =====
    tokens = preprocess_email(text)

    # ===== 2. WORD2VEC =====
    vectors = tokens_to_vectors(tokens, w2v, MAX_LEN, DIM)

    # lấy 5 chiều đầu cho UI
    vectors_5d = [vec[:5].tolist() for vec in vectors[:len(tokens)]]

    # ===== 3. MODEL =====
    result = predict_text(text)

    print("📤 Output:", result)

    # ===== RESPONSE =====
    return {
        "text": text,
        "tokens": tokens,
        "vectors": vectors_5d,
        "prob": result["prob"],
        "label": result["label"]
    }