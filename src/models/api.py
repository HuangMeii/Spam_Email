from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models.predict import predict_text

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
# SCHEMA
# =========================
class InputText(BaseModel):
    text: str


# =========================
# ROOT TEST
# =========================
@app.get("/")
def home():
    return {"msg": "API is running 🚀"}


# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict_api(data: InputText):
    return predict_text(data.text)
