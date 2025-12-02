import os
import json
import torch
import torchvision
from PIL import Image
from pathlib import Path
from torch import optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from create_dataset import create_faster_rcnn_dataset

NUM_EPOCHS = 2
MODELS_ROOT = Path(os.getenv("BOT_MODELS_ROOT"))
OUTPUT_PATH = Path("output")
OUTPUT_FILE_NAME = "results.json"
TRAINING_MONITORING = True
BATCH_SIZE = 8

class PascalVoc(Dataset):
    def __init__(self, path, mode="train", transform=None):
        super().__init__()
        self.transform = transform
        self.images_path = path / "images" / mode
        self.labels_path = path / "labels" / mode
        self.images = os.listdir(self.images_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images_path / self.images[idx]
        image = Image.open(image_path).convert("RGB")

        w, h = image.size
        boxes, labels = [], []

        label_path = self.labels_path / (image_path.stem + ".txt")

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    cid, x, y, bw, bh = map(float, line.split())
                    cid = int(cid)

                    x_min = (x - bw/2) * w
                    y_min = (y - bh/2) * h
                    x_max = (x + bw/2) * w
                    y_max = (y + bh/2) * h

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(cid + 1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target
    
def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for images, targets in tqdm(dataloader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    map_metric = MeanAveragePrecision(iou_thresholds=None, class_metrics=True)

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            map_metric.update(predictions, targets)

    epoch_metrics = map_metric.compute()
    map_metric.reset()

    map50 = epoch_metrics['map_50'].item()
    map50_95 = epoch_metrics['map'].item()
    precision = epoch_metrics.get('map_per_class', torch.tensor([0.0])).mean().item()
    recall = epoch_metrics.get('mar_1', torch.tensor([0.0])).mean().item()

    map_metric.reset()

    return map50, map50_95, precision, recall

num_epochs = NUM_EPOCHS
output_path = OUTPUT_PATH
output_file_name = OUTPUT_FILE_NAME
training_monitoring = TRAINING_MONITORING
batch_size = BATCH_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes=21).to(device)

train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

script_path = "./download_voc.sh"
data_dir = "datasets/VOCdevkit"
save_dir = "datasets/VOC"
create_faster_rcnn_dataset(script_path, data_dir, save_dir)

path = Path("Faster_RCNN_Dataset")
train_dataset = PascalVoc(path, mode="train", transform=train_transform)
val_dataset = PascalVoc(path, mode="test", transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

result_metrics = []

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(num_epochs):
    metrics = {}
    print(f'Epoch {epoch+1}/{num_epochs}')

    train_loss = train_epoch(model, train_loader, optimizer, device)
    if training_monitoring:
        print(f'Train Loss: {train_loss:.4f}')

    epoch_metrics = validate(model, val_loader, device)
    if training_monitoring:
        print(f"  - mAP50: {epoch_metrics[0]:.4f}")
        print(f"  - mAP50-95: {epoch_metrics[1]:.4f}")
        print(f"  - Precision: {epoch_metrics[2]:.4f}")
        print(f"  - Recall: {epoch_metrics[3]:.4f}")
        print('-' * 50) 

    metrics["mAP50"] = epoch_metrics[0]
    metrics["mAP50-95"] = epoch_metrics[1]
    metrics["Precision"] = epoch_metrics[2]
    metrics["Recall"] = epoch_metrics[3]
    metrics["Loss"] = train_loss
    result_metrics.append(metrics)

    lr_scheduler.step()

os.makedirs(MODELS_ROOT / "faster_rcnn", exist_ok=True)
torch.save(model.state_dict(), MODELS_ROOT / "faster_rcnn" / 'faster_rcnn_pascal_voc.pth')

os.makedirs(output_path, exist_ok=True)
with open(output_path / output_file_name, "w", encoding="utf-8") as f:
    json.dump(result_metrics, f, indent=4)

print("The training has been successfully completed")