from fastapi import FastAPI, UploadFile, File
from models.custom_unet import detect_objects

app = FastAPI()

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = detect_objects(image_bytes)
    return {"result": result.hex()}  # передаём байты в hex для совместимости
