import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

MODELS_ROOT = Path(os.getenv("BOT_MODELS_ROOT"))

model = YOLO("yolo11m.yaml")
results = model.train(data="VOC.yaml", epochs=40, imgsz=640, batch=32)

results_csv = "runs/detect/train/results.csv"
df = pd.read_csv(results_csv)

plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
plt.xlabel("Epoch")
plt.ylabel("mAP50")
plt.title("mAP50 during training")
plt.legend()
plt.grid()
plt.show()

os.makedirs(MODELS_ROOT / "yolo")
shutil.move("runs/detect/train/weights/best.pt", MODELS_ROOT / "yolo" / "yolo.pt")