from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from pycaret.regression import load_model, predict_model

app = FastAPI()
model = load_model("outputs/best_model_pycaret")  # caminho do seu modelo

class InputData(BaseModel):
    X: float
    Y: float
    Z: float

@app.post("/predict")

def predict(data: InputData):
    input_df = pd.DataFrame([{
        "X": data.X,
        "Y": data.Y,
        "Z": data.Z,
        "Propriedade": 0  # dummy placeholder, será ignorado na inferência
    }])
    
    prediction = predict_model(model, data=input_df)
    predicted_value = prediction["prediction_label"].iloc[0]
    
    return {"predicted_property": predicted_value}
