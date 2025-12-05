import os
import random
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
from models.yolo.yolo import yolo_detect
from models.groundingdino.groundingdino import groundingdino_detect
from models.detr.detr import detr_detect
from ultralytics import YOLO

MODELS_ROOT = Path(os.getenv("BOT_MODELS_ROOT"))
app = FastAPI()

@app.post("/process/")
async def process_image(
    file: UploadFile = File(...),
    model: str = Form("yolo"),
    description: str = Form("")
):
    image_bytes = await file.read()

    if model == "yolo":
        mdl = YOLO(MODELS_ROOT / "yolo" / "yolo.pt")
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(mdl.names))]
        result = yolo_detect(image_bytes, mdl, colors)

    elif model == "gd":
        if not description:
            description = "object"
        result = groundingdino_detect(image_bytes, description)

    elif model == "rcnn":
        # TODO: добавить реализацию RCNN
        result = image_bytes

    elif model == "detr":
        result = detr_detect(image_bytes)

    return {"result": result.hex()}