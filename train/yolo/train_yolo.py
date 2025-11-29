import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="VOC.yaml", epochs=40, imgsz=640)

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