import cv2
import io
from typing import Any
import numpy as np
from PIL import Image

def yolo_detect(image_bytes: str, model: Any, colors: list[list[int]]) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    cv2.imwrite("input.png", image)
    results = model.predict(image, conf=0.2)
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
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    output_buffer = io.BytesIO()
    image.save(output_buffer, format="JPEG")
    
    return output_buffer.getvalue()