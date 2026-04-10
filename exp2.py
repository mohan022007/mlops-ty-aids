# ============================================================
#   EXPERIMENT 2 — FastAPI Backend for Model Inference
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# --- Load the optimized model ---
model = joblib.load(r"C:\Users\MANMOHAN\OneDrive\Desktop\codes\AI or ML\MLops\random_forest_most_optimized.pkl")

# --- Initialize FastAPI app ---
app = FastAPI(title="Heart Disease Prediction API")


# --- Request Schema (what the user sends) ---
class PredictRequest(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalch: float
    exang: int
    oldpeak: float


# --- Response Schema (what the API returns) ---
class PredictResponse(BaseModel):
    prediction: int
    result: str


# --- Root endpoint ---
@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running!"}


# --- Predict endpoint ---
@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    input_data = np.array([[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalch,
        data.exang,
        data.oldpeak
    ]])

    prediction = model.predict(input_data)[0]
    result = "Disease Detected" if prediction == 1 else "No Disease"

    return PredictResponse(prediction=int(prediction), result=result)