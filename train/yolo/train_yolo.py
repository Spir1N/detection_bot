from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data="VOC.yaml", epochs=40, imgsz=640)