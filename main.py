from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from utils.data_handler import save_uploaded_file
from utils.model_handler import train_model, predict_with_model
import os
import logging
import pandas as pd

app = FastAPI()

# Global variables
UPLOAD_DIR = "./static/"
MODEL_DIR = "./model_store/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# Request model for prediction
class PredictRequest(BaseModel):
    UDI: int
    Type: str
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: int
    Torque: float
    Tool_wear: int
    TWF: int
    HDF: int
    PWF: int
    OSF: int
    RNF: int

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    await save_uploaded_file(file, filepath)
    logging.info(f"File uploaded to {filepath}")
    return {"message": "File uploaded successfully", "filepath": filepath}

@app.post("/train")
def train():
    filepath = os.path.join(UPLOAD_DIR, "uploaded_data.csv")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="No data file found. Please upload first.")
    
    metrics = train_model(filepath, os.path.join(MODEL_DIR, "model.joblib"))
    logging.info(f"Model trained and saved to {os.path.join(MODEL_DIR, 'model.joblib')}")
    return {"message": "Model trained successfully", "metrics": metrics}

@app.post("/predict")
def predict(request: PredictRequest):
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="No trained model found. Please train the model first.")
    
    # Input data for prediction
    input_data = pd.DataFrame([{
        'UDI': request.UDI,
        'Type': request.Type,
        'Air temperature [K]': request.Air_temperature,
        'Process temperature [K]': request.Process_temperature,
        'Rotational speed [rpm]': request.Rotational_speed,
        'Torque [Nm]': request.Torque,
        'Tool wear [min]': request.Tool_wear,
        'TWF': request.TWF,
        'HDF': request.HDF,
        'PWF': request.PWF,
        'OSF': request.OSF,
        'RNF': request.RNF
    }])
    
    prediction, confidence = predict_with_model(model_path, input_data)
    logging.info(f"Prediction: {prediction}, Confidence: {confidence}")
    return {"Downtime": "Yes" if prediction else "No", "Confidence": round(confidence, 2)}