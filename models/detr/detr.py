import os
import io
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrImageProcessor, DetrForObjectDetection

MODELS_ROOT = Path(os.getenv("BOT_MODELS_ROOT"))
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
COLORS = [
    "red", "green", "blue", "yellow", "purple", "orange", "cyan",
    "magenta", "lime", "pink", "white"
]

def get_detr_model(best_model_path, device):
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(VOC_CLASSES),
        ignore_mismatched_sizes=True
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval() 
    return model, image_processor

def infer_image(model, processor, image_bytes, score_threshold=0.5, device="cpu"):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = image.size
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits.softmax(-1)[0, :, :-1]
    scores, labels = probs.max(-1)

    keep = scores > score_threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = outputs.pred_boxes[0][keep]

    cx = boxes[:, 0] * orig_w
    cy = boxes[:, 1] * orig_h
    w = boxes[:, 2] * orig_w
    h = boxes[:, 3] * orig_h

    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2

    xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1)

    return xyxy.cpu(), labels.cpu(), scores.cpu(), image

def draw_boxes(image, boxes, labels, scores, output_path="result.jpg"):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        label = VOC_CLASSES[labels[i]]
        score = scores[i].item()

        color = COLORS[labels[i] % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        text = f"{label}: {score:.2f}"
        
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
        draw.text((x1, y1 - text_h), text, fill="black", font=font)

    image.save(output_path)
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="JPEG")

    print(f"Saved result to: {output_path}")

    return output_buffer.getvalue()

def detr_detect(image_bytes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = MODELS_ROOT / "detr" / "best_detr_voc.pth"
    model, processor = get_detr_model(best_model_path, device)
    boxes, labels, scores, img = infer_image(
        model, processor,
        image_bytes=image_bytes,
        score_threshold=0.5,
        device=device
    )

    return draw_boxes(img, boxes, labels, scores, output_path="predicted.jpg")