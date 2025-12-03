# src/serve.py
import os
import joblib
import numpy as np
import logging
from fastapi import FastAPI, HTTPException

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve")

app = FastAPI(title='ESG Test App - Model Serve')

MODEL_NAME = os.environ.get('MODEL_NAME', 'iris_rf_model')
MODEL_STAGE = os.environ.get('MODEL_STAGE', 'Production')
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')

# Try local fallback model first
local_model_path = os.path.join('artifacts', 'model.joblib')
model = None
if os.path.exists(local_model_path):
    try:
        model = joblib.load(local_model_path)
        logger.info(f"Loaded local model from {local_model_path}")
    except Exception as e:
        logger.exception(f"Failed to load local model: {e}")
        model = None
else:
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        logger.info(f"Loaded model from MLflow: models:/{MODEL_NAME}/{MODEL_STAGE}")
    except Exception as e:
        logger.exception(f"Warning: failed to load model from MLflow: {e}")
        model = None

class Input(dict):
    # simple duck-typing Pydantic not required for health check
    pass

@app.get("/")
def root():
    return {'status':'ok','model_loaded': model is not None}

@app.get("/health")
def health():
    # simple health: OK if server up (model optional)
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(inp: dict):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    try:
        arr = np.array(inp.get("features", [])).reshape(1, -1)
        pred = model.predict(arr).tolist()
        return {'prediction': pred}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))
