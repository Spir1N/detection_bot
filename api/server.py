import random
from fastapi import FastAPI, UploadFile, File
from models.custom_unet import detect_objects
from models.yolo import yolo_detect
from ultralytics import YOLO

app = FastAPI()

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    model = YOLO("yolov8x-seg.pt")
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(model.names))]
    image_bytes = await file.read()
    result = yolo_detect(image_bytes, model, colors)
    return {"result": result.hex()}