# src/serve.py
import os, joblib, numpy as np, mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title='ESG Test App - Model Serve')

MODEL_NAME = os.environ.get('MODEL_NAME', 'iris_rf_model')
MODEL_STAGE = os.environ.get('MODEL_STAGE', 'Production')
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')

# Try local fallback model first
local_model_path = os.path.join('artifacts', 'model.joblib')
model = None
if os.path.exists(local_model_path):
    model = joblib.load(local_model_path)
else:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except Exception as e:
        print('Warning: failed to load model from MLflow:', e)
        model = None

class Input(BaseModel):
    features: list

@app.get('/')
def root():
    return {'status':'ok','model_loaded': model is not None}

@app.post('/predict')
def predict(inp: Input):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    try:
        arr = np.array(inp.features).reshape(1, -1)
        pred = model.predict(arr).tolist()
        return {'prediction': pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
