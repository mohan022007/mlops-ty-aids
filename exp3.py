import logging
import numpy as np
import joblib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from datetime import datetime

# LOGGING SETUP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),   # saves logs to a file
        logging.StreamHandler()           # also prints to terminal
    ]
)

logger = logging.getLogger(__name__)

# LOAD MODEL

try:
    model = joblib.load(r"C:\Users\MANMOHAN\OneDrive\Desktop\codes\AI or ML\MLops\random_forest_most_optimized.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# APP INIT

app = FastAPI(title="Heart Disease Prediction API — Exp 3")

# SCHEMAS


class PredictRequest(BaseModel):
    age: float      = Field(..., gt=0, lt=120, description="Age in years")
    sex: int        = Field(..., ge=0, le=1,   description="0 = Female, 1 = Male")
    cp: int         = Field(..., ge=0, le=3,   description="Chest pain type (0-3)")
    trestbps: float = Field(..., gt=0,         description="Resting blood pressure")
    chol: float     = Field(..., gt=0,         description="Cholesterol level")
    fbs: int        = Field(..., ge=0, le=1,   description="Fasting blood sugar > 120 (0/1)")
    restecg: int    = Field(..., ge=0, le=2,   description="Resting ECG result (0-2)")
    thalch: float   = Field(..., gt=0,         description="Max heart rate achieved")
    exang: int      = Field(..., ge=0, le=1,   description="Exercise induced angina (0/1)")
    oldpeak: float  = Field(..., ge=0,         description="ST depression")

class PredictResponse(BaseModel):
    prediction: int
    result: str
    timestamp: str

# EXCEPTION HANDLERS

# handles invalid input (wrong types, missing fields, out of range values)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error | {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid input",
            "details": exc.errors()
        }
    )

# handles any unexpected server errors
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error | {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc)
        }
    )


# ENDPOINTS

@app.get("/")
def root():
    logger.info("Root endpoint hit")
    return {"message": "Heart Disease Prediction API is running!"}


@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest, request: Request):
    logger.info(f"Incoming request | IP: {request.client.host} | Data: {data.dict()}")

    try:
        input_data = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalch, data.exang, data.oldpeak
        ]])

        prediction = model.predict(input_data)[0]
        result = "Disease Detected" if prediction == 1 else "No Disease"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"Prediction made | Result: {result} | Timestamp: {timestamp}")

        return PredictResponse(
            prediction=int(prediction),
            result=result,
            timestamp=timestamp
        )

    except Exception as e:
        logger.error(f"Prediction error | {str(e)}")
        raise