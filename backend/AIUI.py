import time
import uuid
import random
from typing import Dict, List

import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# ------------------------------
# Define a simple AI model class.
# In a real system, this might wrap an actual machine learning model.
# ------------------------------
class AIModel:
    def __init__(self, model_id: str, name: str, version: str, metrics: Dict = None):
        self.model_id = model_id
        self.name = name
        self.version = version
        self.metrics = metrics if metrics is not None else {}
        self.deployed = False

# ------------------------------
# ModelManager keeps track of all models.
# ------------------------------
class ModelManager:
    def __init__(self):
        self.models: Dict[str, AIModel] = {}
        # Pre-populate with some dummy models
        self.add_model("Model A", "1.0")
        self.add_model("Model B", "1.0")

    def add_model(self, name: str, version: str) -> AIModel:
        model_id = str(uuid.uuid4())
        model = AIModel(model_id, name, version)
        self.models[model_id] = model
        return model

    def list_models(self) -> List[AIModel]:
        return list(self.models.values())

    def get_model(self, model_id: str) -> AIModel:
        return self.models.get(model_id)

    def deploy_model(self, model_id: str) -> AIModel:
        model = self.get_model(model_id)
        if model:
            # Here you might load the model into memory or start an inference server.
            model.deployed = True
            return model
        else:
            raise ValueError("Model not found")

    def fine_tune_model(self, model_id: str, tuning_params: Dict) -> AIModel:
        model = self.get_model(model_id)
        if not model:
            raise ValueError("Model not found")
        # Simulate a fine-tuning (training) process.
        print(f"Starting fine-tuning for model {model.name}...")
        time.sleep(5)  # simulate training time
        # Simulate updating model metrics after fine-tuning.
        model.metrics["accuracy"] = round(random.uniform(0.80, 0.99), 3)
        # Optionally update version information.
        new_version = tuning_params.get("new_version")
        if new_version:
            model.version = new_version
        print(f"Fine-tuning complete for model {model.name}. New metrics: {model.metrics}")
        return model

    def get_metrics(self, model_id: str) -> Dict:
        model = self.get_model(model_id)
        if model:
            return model.metrics
        else:
            raise ValueError("Model not found")

# ------------------------------
# FastAPI Application Setup
# ------------------------------
app = FastAPI()
manager = ModelManager()

# ------------------------------
# Pydantic model for fine-tuning request data.
# ------------------------------
class FineTuneRequest(BaseModel):
    model_id: str
    tuning_params: Dict[str, str] = {}

# ------------------------------
# Endpoint to list all models.
# ------------------------------
@app.get("/models", response_model=List[Dict])
def list_models():
    models = manager.list_models()
    # Return a list of models with relevant details.
    return [
        {
            "model_id": m.model_id,
            "name": m.name,
            "version": m.version,
            "deployed": m.deployed,
            "metrics": m.metrics,
        }
        for m in models
    ]

# ------------------------------
# Endpoint to get details for a specific model.
# ------------------------------
@app.get("/models/{model_id}", response_model=Dict)
def get_model(model_id: str):
    model = manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "model_id": model.model_id,
        "name": model.name,
        "version": model.version,
        "deployed": model.deployed,
        "metrics": model.metrics,
    }

# ------------------------------
# Endpoint to deploy a model.
# ------------------------------
@app.post("/deploy/{model_id}", response_model=Dict)
def deploy_model(model_id: str):
    try:
        model = manager.deploy_model(model_id)
        return {
            "message": f"Model '{model.name}' deployed successfully.",
            "model": {
                "model_id": model.model_id,
                "name": model.name,
                "version": model.version,
                "deployed": model.deployed,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ------------------------------
# Endpoint to start fine-tuning a model.
# The fine-tuning process is started as a background task.
# ------------------------------
@app.post("/finetune", response_model=Dict)
def finetune_model(request: FineTuneRequest, background_tasks: BackgroundTasks):
    def fine_tune_task():
        try:
            manager.fine_tune_model(request.model_id, request.tuning_params)
        except Exception as e:
            # In a real system, you would log this exception.
            print(f"Error during fine-tuning: {e}")

    background_tasks.add_task(fine_tune_task)
    return {"message": "Fine-tuning started in background."}

# ------------------------------
# Endpoint to retrieve metrics for a given model.
# ------------------------------
@app.get("/metrics/{model_id}", response_model=Dict)
def get_metrics(model_id: str):
    try:
        metrics = manager.get_metrics(model_id)
        return {"model_id": model_id, "metrics": metrics}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ------------------------------
# A simple system monitoring endpoint.
# Returns CPU and memory usage.
# ------------------------------
@app.get("/monitor", response_model=Dict)
def monitor():
    cpu_usage = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    return {
        "cpu_usage": cpu_usage,
        "memory": {
            "total": mem.total,
            "available": mem.available,
            "percent": mem.percent,
        },
    }
