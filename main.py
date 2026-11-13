from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
app = FastAPI(title="Machine Failure Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "machine_failure_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

class MachineData(BaseModel):
    Type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
@app.get("/")
def home():
    return {"message": "Machine Failure Prediction API is running"}
@app.post("/predict")
def predict(data: MachineData):
    try:
        input_data = pd.DataFrame([{
            "Type": data.Type,
            "Air temperature [K]": data.air_temperature,
            "Process temperature [K]": data.process_temperature,
            "Rotational speed [rpm]": data.rotational_speed,
            "Torque [Nm]": data.torque,
            "Tool wear [min]": data.tool_wear
        }])
        input_data = pd.get_dummies(input_data, columns=["Type"])

        input_data = input_data.reindex(
            columns=model_columns,
            fill_value=0
        )

        prediction = int(model.predict(input_data)[0])

        if prediction == 1:
            result = "Machine Failure"
        else:
            result = "No Machine Failure"

        return {
            "prediction": prediction,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
