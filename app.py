from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Machine Failure Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "machine_failure_model.pkl"))
model_columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))

class MachineInput(BaseModel):
    machine_type: str
    air_temp: float
    process_temp: float
    rotational_speed: float
    torque: float
    tool_wear: float

@app.get("/")
def home():
    return {"message": "Machine Failure Prediction API is running"}

@app.post("/predict")
def predict(data: MachineInput):
    try:
        input_data = {
            "Air temperature [K]": data.air_temp,
            "Process temperature [K]": data.process_temp,
            "Rotational speed [rpm]": data.rotational_speed,
            "Torque [Nm]": data.torque,
            "Tool wear [min]": data.tool_wear
        }
        df = pd.DataFrame(0, index=[0], columns=model_columns)
        for column, value in input_data.items():
            if column in df.columns:
                df[column] = value
        type_column = "Type_" + data.machine_type
        if type_column in df.columns:
            df[type_column] = 1
        else:
            raise HTTPException(status_code=400, detail="Invalid machine type")
        prediction = int(model.predict(df)[0])
        result = "Machine Failure Predicted" if prediction == 1 else "No Machine Failure Predicted"
        return {"prediction": prediction, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
