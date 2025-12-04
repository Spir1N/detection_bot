import os
import random
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from models.yolo import yolo_detect
from models.groundingdino.groundingdino import groundingdino_detect
from ultralytics import YOLO

MODELS_ROOT = Path(os.getenv("BOT_MODELS_ROOT"))
app = FastAPI()
MODELS = ["yolo", 'gd']
model = MODELS[1]

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    model = 'gd'
    image_bytes = await file.read()
    if model == "yolo":
        model = YOLO(MODELS_ROOT / "yolo" / "yolo.pt")
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(model.names))]
        result = yolo_detect(image_bytes, model, colors)
    elif model == "gd":
        description = "coach"
        result = groundingdino_detect(image_bytes, description)
    
    return {"result": result.hex()}