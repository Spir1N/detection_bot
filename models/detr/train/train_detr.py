import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import DetrImageProcessor, DetrForObjectDetection

from create_dataset import create_dataset

NUM_EPOCHS = 10
MODELS_ROOT = Path(os.getenv("BOT_MODELS_ROOT"))
OUTPUT_PATH = Path("output")
OUTPUT_FILE_NAME = "results.json"
TRAINING_MONITORING = True
BATCH_SIZE = 8
PRETRAINED = True
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
class_to_id = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

def convert_voc_to_detr_format(target):
    boxes = []
    labels = []
    for obj in target['annotation']['object']:
        bbox = obj['bndbox']
        xmin = float(bbox['xmin'])
        ymin = float(bbox['ymin'])
        xmax = float(bbox['xmax'])
        ymax = float(bbox['ymax'])
        boxes.append([xmin, ymin, xmax, ymax])

        class_name = obj['name']
        if class_name not in class_to_id:
            raise ValueError(f"Неизвестный класс: {class_name}")
        labels.append(class_to_id[class_name])

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    return {'boxes': boxes, 'labels': labels}


class VOCDetrDataset(torch.utils.data.Dataset):
    def __init__(self, voc_dataset, image_processor, target_size=800):
        self.voc_dataset = voc_dataset
        self.image_processor = image_processor
        self.target_size = target_size

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        image, target_voc = self.voc_dataset[idx]
        orig_w, orig_h = image.size

        target = convert_voc_to_detr_format(target_voc)
        assert target['labels'].max().item() < len(VOC_CLASSES), f"Найден недопустимый класс: {target['labels'].max().item()}"
        assert target['labels'].min().item() >= 0, f"Найден отрицательный класс: {target['labels'].min().item()}"

        inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            size={"height": self.target_size, "width": self.target_size}
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        boxes = target['boxes'].clone()
        boxes[:, [0, 2]] /= orig_w
        boxes[:, [1, 3]] /= orig_h

        xmin, ymin, xmax, ymax = boxes.unbind(1)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        boxes_cxcywh = torch.stack([cx, cy, w, h], dim=1)

        detr_target = {
            "class_labels": target['labels'].to(torch.int64),
            "boxes": boxes_cxcywh.to(torch.float32),
            "orig_size": torch.tensor([orig_w, orig_h], dtype=torch.float32)
        }

        return inputs, detr_target

def train(model, dataloader, optimizer, device):
    model.train()
    total_train_loss = 0.0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        pixel_values = torch.stack([item[0]['pixel_values'] for item in batch]).to(device)
        pixel_mask = None
        if 'pixel_mask' in batch[0][0]:
            pixel_mask = torch.stack([item[0]['pixel_mask'] for item in batch]).to(device)

        targets = []
        for item in batch:
            t = item[1]
            targets.append({
                "class_labels": t["class_labels"].to(device),
                "boxes": t["boxes"].to(device),
            })

        if pixel_mask is not None:
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=targets)
        else:
            outputs = model(pixel_values=pixel_values, labels=targets)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(dataloader)

    return avg_train_loss

def validate(model, dataloader, device):
    model.eval()
    total_val_loss = 0.0
    map_metric.reset()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = torch.stack([item[0]['pixel_values'] for item in batch]).to(device)
            pixel_mask = None
            if 'pixel_mask' in batch[0][0]:
                pixel_mask = torch.stack([item[0]['pixel_mask'] for item in batch]).to(device)

            targets_for_model = []
            orig_sizes = []
            for item in batch:
                t = item[1]
                targets_for_model.append({
                    "class_labels": t["class_labels"].to(device),
                    "boxes": t["boxes"].to(device),
                })
                orig_sizes.append(t["orig_size"].cpu())

            if pixel_mask is not None:
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=targets_for_model)
            else:
                outputs = model(pixel_values=pixel_values, labels=targets_for_model)

            total_val_loss += outputs.loss.item()
            prob = F.softmax(outputs.logits, dim=-1)
            scores, labels = prob[..., :-1].max(-1)

            pred_boxes_cxcywh = outputs.pred_boxes.cpu()
            batch_preds = []
            batch_targets_for_metric = []

            for b_idx in range(pred_boxes_cxcywh.shape[0]):
                orig_w, orig_h = orig_sizes[b_idx].tolist()
                b_scores = scores[b_idx].cpu()
                b_labels = labels[b_idx].cpu()
                b_boxes = pred_boxes_cxcywh[b_idx]

                keep = b_scores > 0.5
                if keep.sum() == 0:
                    preds = {"boxes": torch.zeros((0,4), dtype=torch.float32),
                             "scores": torch.zeros((0,), dtype=torch.float32),
                             "labels": torch.zeros((0,), dtype=torch.int64)}
                else:
                    kept_scores = b_scores[keep].to(torch.float32)
                    kept_labels = b_labels[keep].to(torch.int64)
                    kept_boxes = b_boxes[keep]
                    cx = kept_boxes[:, 0] * orig_w
                    cy = kept_boxes[:, 1] * orig_h
                    w = kept_boxes[:, 2] * orig_w
                    h = kept_boxes[:, 3] * orig_h
                    xmin = cx - 0.5 * w
                    ymin = cy - 0.5 * h
                    xmax = cx + 0.5 * w
                    ymax = cy + 0.5 * h
                    xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=1).to(torch.float32)

                    preds = {
                        "boxes": xyxy,
                        "scores": kept_scores,
                        "labels": kept_labels
                    }

                batch_preds.append(preds)

                t_boxes_cxcywh = targets_for_model[b_idx]["boxes"].cpu()
                t_labels = targets_for_model[b_idx]["class_labels"].cpu()
                tcx = t_boxes_cxcywh[:, 0] * orig_w
                tcy = t_boxes_cxcywh[:, 1] * orig_h
                tw = t_boxes_cxcywh[:, 2] * orig_w
                th = t_boxes_cxcywh[:, 3] * orig_h
                txmin = tcx - 0.5 * tw
                tymin = tcy - 0.5 * th
                txmax = tcx + 0.5 * tw
                tymax = tcy + 0.5 * th
                t_xyxy = torch.stack([txmin, tymin, txmax, tymax], dim=1).to(torch.float32)

                batch_targets_for_metric.append({
                    "boxes": t_xyxy,
                    "labels": t_labels.to(torch.int64)
                })

            map_metric.update(batch_preds, batch_targets_for_metric)

    epoch_metrics = map_metric.compute()
    map50 = epoch_metrics.get('map_50', torch.tensor(0.0)).item()
    map50_95 = epoch_metrics.get('map', torch.tensor(0.0)).item()
    avg_val_loss = total_val_loss / len(val_loader)

    return map50, map50_95, avg_val_loss

num_epochs = NUM_EPOCHS
output_path = OUTPUT_PATH
output_file_name = OUTPUT_FILE_NAME
batch_size = BATCH_SIZE
pretrained = PRETRAINED

if not os.path.exists("datasets"):
    script_path = "./download_voc.sh"
    create_dataset(script_path)

data_dir = "./datasets/data" 
train_dataset = datasets.VOCDetection(
    root=data_dir, year='2012', image_set='train', download=False
)

val_dataset = datasets.VOCDetection(
    root=data_dir, year='2012', image_set='val', download=False
)

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
train_detr = VOCDetrDataset(train_dataset, image_processor, target_size=800)
val_detr = VOCDetrDataset(val_dataset, image_processor, target_size=800)

train_loader = DataLoader(
    dataset=train_detr, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=lambda x: x
    )
val_loader = DataLoader(
    val_detr, 
    batch_size=1, 
    shuffle=False, 
    collate_fn=lambda x: x
    )

best_val_loss = float('inf')
best_model_path = MODELS_ROOT / "detr" / "best_detr_voc.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(VOC_CLASSES),
    ignore_mismatched_sizes=True
).to(device)
if pretrained:
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

map_metric = MeanAveragePrecision()
result_metrics = []
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    metrics = {}
    print(f'Epoch {epoch+1}/{num_epochs}')

    train_loss = train(model, train_loader, optimizer, device)
    map50, map50_95, avg_val_loss = validate(model, val_loader, device)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}")
    print(50 * "-")

    metrics["mAP50"] = map50
    metrics["mAP50-95"] = map50_95
    metrics["Train loss"] = train_loss
    metrics["Loss"] = avg_val_loss
    result_metrics.append(metrics)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)

os.makedirs(MODELS_ROOT / "detr", exist_ok=True)
torch.save(model.state_dict(), best_model_path)

os.makedirs(output_path, exist_ok=True)
with open(output_path / output_file_name, "w", encoding="utf-8") as f:
    json.dump(result_metrics, f, indent=4)

print("The training has been successfully completed")