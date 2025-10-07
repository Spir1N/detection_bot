from ultralytics import YOLO
import cv2, random
from typing import Any
import numpy as np

def yolo_detect(image_bytes: str, model: Any, color: list[list[int]]) -> np.ndarray:
    image = cv2.imread(image_bytes)
    results = model.predict(source=image)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            color = colors[cls]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            text = f'{label} {confidence:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            cv2.rectangle(image, (x1, y1 - text_height - 14), (x1 + text_width, y1), color, -1)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imwrite("output.png", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image