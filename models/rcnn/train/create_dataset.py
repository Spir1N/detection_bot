import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_ID = {name: i for i, name in enumerate(VOC_CLASSES)}

def download_data(script_path):
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Файл {script_path} не найден.")
    
    print("Uploading a dataset...")
    os.chmod(script_path, 0o755)
    subprocess.run(['bash', script_path])
    print("Dataset is loaded")

def convert_bbox_voc_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_image_and_annotations(image_path, annotation_path, output_labels_dir):
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
    except FileNotFoundError:
        print(f"Warning: Image file not found at {image_path}. Skipping.")
        return
    except Exception as e:
        print(f"Warning: Could not open image {image_path}: {e}. Skipping.")
        return

    tree = ET.parse(annotation_path)
    root = tree.getroot()

    yolo_annotations = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in CLASS_TO_ID:
            print(f"Warning: Class '{class_name}' not found in VOC_CLASSES. Skipping object in {annotation_path}.")
            continue

        class_id = CLASS_TO_ID[class_name]
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        b = (xmin, xmax, ymin, ymax)
        yolo_box = convert_bbox_voc_to_yolo((img_width, img_height), b)
        yolo_annotations.append(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_label_path = os.path.join(output_labels_dir, f"{base_filename}.txt")
    with open(output_label_path, 'w') as f:
        for line in yolo_annotations:
            f.write(line + '\n')

def convert_voc_to_yolo_custom(voc_root_dir, yolo_output_dir, use2007=True, use2012=True):
    os.makedirs(yolo_output_dir, exist_ok=True)

    train_images_dir = os.path.join(yolo_output_dir, 'images', 'train')
    train_labels_dir = os.path.join(yolo_output_dir, 'labels', 'train')
    val_images_dir = os.path.join(yolo_output_dir, 'images', 'val')
    val_labels_dir = os.path.join(yolo_output_dir, 'labels', 'val')
    test_images_dir = os.path.join(yolo_output_dir, 'images', 'test')
    test_labels_dir = os.path.join(yolo_output_dir, 'labels', 'test')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    sets_to_process = []
    if use2007:
        sets_to_process.append(('2007', 'trainval', train_images_dir, train_labels_dir))
        sets_to_process.append(('2007', 'test', test_images_dir, test_labels_dir))
    if use2012:
        sets_to_process.append(('2012', 'trainval', train_images_dir, train_labels_dir))

    for year, image_set_name, current_images_dir, current_labels_dir in sets_to_process:
        print(f"Processing VOC{year}_{image_set_name}...")
        
        image_set_file_path = os.path.join(voc_root_dir, f'VOC{year}', 'ImageSets', 'Main', f'{image_set_name}.txt')
        if not os.path.exists(image_set_file_path):
             print(f"Warning: Image set file not found at {image_set_file_path}. Skipping this set.")
             continue

        with open(image_set_file_path, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]

        for image_id in image_ids:
            voc_image_path = os.path.join(voc_root_dir, f'VOC{year}', 'JPEGImages', f'{image_id}.jpg')
            voc_annotation_path = os.path.join(voc_root_dir, f'VOC{year}', 'Annotations', f'{image_id}.xml')

            if not os.path.exists(voc_image_path):
                print(f"Warning: Image file not found for {image_id}. Skipping.")
                continue
            if not os.path.exists(voc_annotation_path):
                print(f"Warning: Annotation file not found for {image_id}. Skipping.")
                continue

            shutil.copy(voc_image_path, current_images_dir)
            process_image_and_annotations(voc_image_path, voc_annotation_path, current_labels_dir)

def copy_directory_contents(source_dirs, target_dir):
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if source_path.exists() and source_path.is_dir():
            for item in source_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_dir / item.name)
                elif item.is_dir():
                    for file in item.rglob('*'):
                        if file.is_file():
                            relative_path = file.relative_to(source_path)
                            target_path = target_dir / relative_path
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(file, target_path)

def create_faster_rcnn_dataset(script_path, data_dir, save_dir):
    download_data(
        script_path=script_path
        )
    convert_voc_to_yolo_custom(
        voc_root_dir=data_dir,
        yolo_output_dir=save_dir,
        use2007=True,
        use2012=True
    )

    print("Creating final dataset")
    
    main_dir = Path("Faster_RCNN_Dataset")
    main_dir.mkdir(exist_ok=True)

    images_dir = main_dir / "images"
    labels_dir = main_dir / "labels"

    images_train_dir = images_dir / "train"
    images_test_dir = images_dir / "test"
    labels_train_dir = labels_dir / "train"
    labels_test_dir = labels_dir / "test"

    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_test_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_test_dir.mkdir(parents=True, exist_ok=True)

    images_train_sources = [
        "datasets/VOC/images/train",
        "datasets/VOC/images/val",
    ]
    copy_directory_contents(images_train_sources, images_train_dir)

    labels_train_sources = [
        "datasets/VOC/labels/train",
        "datasets/VOC/labels/val",
    ]
    copy_directory_contents(labels_train_sources, labels_train_dir)

    images_test_sources = ["datasets/VOC/images/test"]
    copy_directory_contents(images_test_sources, images_test_dir)

    labels_test_sources = ["datasets/VOC/labels/test"]
    copy_directory_contents(labels_test_sources, labels_test_dir)
    
    def count_files(directory):
        return sum(1 for _ in Path(directory).rglob('*') if _.is_file())

    print(f"Files in images/train: {count_files(images_train_dir)}")
    print(f"Files in labels/train: {count_files(labels_train_dir)}")
    print(f"Files in images/test: {count_files(images_test_dir)}")
    print(f"Files in labels/test: {count_files(labels_test_dir)}")