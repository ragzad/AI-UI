import os
import time
import uuid
import random
from typing import Dict, List

import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# This class represents a simple AI model.
class AIModel:
    def __init__(self, model_id: str, name: str, version: str, metrics: Dict = None):
        self.model_id = model_id
        self.name = name
        self.version = version
        self.metrics = metrics or {}
        self.deployed = False

# This class manages our AI models in memory.
class ModelManager:
    def __init__(self):
        self.models: Dict[str, AIModel] = {}
        # Add a couple of test models on startup.
        self.add_model("Model A", "1.0")
        self.add_model("Model B", "1.0")

    def add_model(self, name: str, version: str) -> AIModel:
        new_id = str(uuid.uuid4())
        model = AIModel(new_id, name, version)
        self.models[new_id] = model
        return model

    def list_models(self) -> List[AIModel]:
        return list(self.models.values())

    def get_model(self, model_id: str) -> AIModel:
        return self.models.get(model_id)

    def deploy_model(self, model_id: str) -> AIModel:
        model = self.get_model(model_id)
        if model:
            model.deployed = True
            return model
        raise ValueError("Model not found")

    def fine_tune_model(self, model_id: str, tuning_params: Dict) -> AIModel:
        model = self.get_model(model_id)
        if not model:
            raise ValueError("Model not found")
        # Simulate a fine-tuning process.
        print(f"Starting fine-tuning for {model.name}...")
        time.sleep(5)  # Pretend we're fine-tuning here.
        model.metrics["accuracy"] = round(random.uniform(0.80, 0.99), 3)
        if "new_version" in tuning_params:
            model.version = tuning_params["new_version"]
        print(f"Finished fine-tuning for {model.name}: {model.metrics}")
        return model

    def get_metrics(self, model_id: str) -> Dict:
        model = self.get_model(model_id)
        if model:
            return model.metrics
        raise ValueError("Model not found")

# Initialize the FastAPI app.
app = FastAPI()

# Figure out the absolute path to the frontend folder.
# Our project structure is:
# AI-UI/
# ├── backend/
# │    └── app.py
# └── frontend/
#      └── index.html
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(base_dir, "..", "frontend")

# Mount the static directory so we can serve CSS/JS/images if needed.
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Serve the main HTML page from the frontend folder.
@app.get("/", response_class=FileResponse)
async def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    return FileResponse(index_file)

# Instantiate our model manager.
manager = ModelManager()

# Define the schema for fine-tuning requests.
class FineTuneRequest(BaseModel):
    model_id: str
    tuning_params: Dict[str, str] = {}

# Endpoint to list all models.
@app.get("/models", response_model=List[Dict])
def get_all_models():
    models = manager.list_models()
    return [
        {
            "model_id": model.model_id,
            "name": model.name,
            "version": model.version,
            "deployed": model.deployed,
            "metrics": model.metrics,
        }
        for model in models
    ]

# Endpoint to get details for a single model.
@app.get("/models/{model_id}", response_model=Dict)
def get_model_details(model_id: str):
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

# Endpoint to deploy a model.
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

# Endpoint to start the fine-tuning process in the background.
@app.post("/finetune", response_model=Dict)
def start_fine_tuning(request: FineTuneRequest, background_tasks: BackgroundTasks):
    def background_task():
        try:
            manager.fine_tune_model(request.model_id, request.tuning_params)
        except Exception as err:
            print(f"Error during fine-tuning: {err}")

    background_tasks.add_task(background_task)
    return {"message": "Fine-tuning started in the background."}

# Endpoint to fetch model metrics.
@app.get("/metrics/{model_id}", response_model=Dict)
def get_model_metrics(model_id: str):
    try:
        metrics = manager.get_metrics(model_id)
        return {"model_id": model_id, "metrics": metrics}
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err))

# Endpoint to monitor system resources.
@app.get("/monitor", response_model=Dict)
def system_monitor():
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
