import os
import random
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from models.yolo import yolo_detect
from ultralytics import YOLO

MODELS_ROOT = Path(os.getenv("BOT_STORAGE_ROOT"))
app = FastAPI()

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    model = YOLO(MODELS_ROOT / "yolo" / "yolo.pt")
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(model.names))]
    image_bytes = await file.read()
    result = yolo_detect(image_bytes, model, colors)
    return {"result": result.hex()}