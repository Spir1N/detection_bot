import os
import shutil
import subprocess

def download_data(script_path):
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Файл {script_path} не найден.")
    
    print("Uploading a dataset...")
    os.chmod(script_path, 0o755)
    subprocess.run(['bash', script_path])
    print("Dataset is loaded")

def create_dataset(script_path):
    download_data(script_path=script_path)
    voc_root = "datasets/data/VOCdevkit/VOC2012"
    os.makedirs(voc_root, exist_ok=True)
    train_val_dir = "datasets/pascal-voc-2012-dataset/VOC2012_train_val/VOC2012_train_val"
    for item in os.listdir(train_val_dir):
        src = os.path.join(train_val_dir, item)
        dst = os.path.join(voc_root, item)
        if not os.path.exists(dst):
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)