from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)

FEATURES = ["G1", "G2", "studytime", "failures", "absences"]

app = FastAPI(
    title="Student Performance Prediction API",
    version="1.0.0"
)

class StudentInput(BaseModel):
    G1: float = Field(ge=0, le=20)
    G2: float = Field(ge=0, le=20)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=3)
    absences: int = Field(ge=0, le=93)

@app.get("/")
def home():
    return {"message": "Student Performance API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: StudentInput):
    values = pd.DataFrame(
        [[data.G1, data.G2, data.studytime,
          data.failures, data.absences]],
        columns=FEATURES
    )
    prediction = float(model.predict(values)[0])
    return {
        "predicted_final_grade": round(prediction, 2),
        "model": "RandomForestRegressor"
    }
