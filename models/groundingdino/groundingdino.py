import os
import io
import json
import uuid
import numpy as np
from PIL import Image
from pathlib import Path

from .comfy_utils import comfy_ws, upload_image, get_images

DATA_ROOT = Path(os.getenv("BOT_DATA_ROOT"))

def get_bboxes(image_bytes, ws, client_id, server_address, description):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(module_dir, "groundingdino.json")
    with open(config_path, 'r') as file:
        prompt = json.load(file)

    image = Image.open(io.BytesIO(image_bytes))
    cached_image_dir = DATA_ROOT / "images" / "cached"
    os.makedirs(cached_image_dir, exist_ok=True)
    image_id = uuid.uuid4()
    image_path = cached_image_dir / f"{image_id}.png"
    image.save(image_path)
    upload_image(server_address, image_path, os.path.basename(image_path), "input", True)
    prompt["4"]["inputs"]["prompt"] = description
    prompt["1"]["inputs"]["image"] = os.path.basename(image_path)

    output_image = get_images(prompt, ws, client_id, server_address)
    for node_id in output_image:
        for image_data in output_image[node_id]:
            image = Image.open(io.BytesIO(image_data))
            image = np.array(image)
    
    return image

def groundingdino_detect(image_bytes, description):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(module_dir, "config.json")
    assert os.path.exists(config_path), f"Config not found at: {config_path}"
    with open(config_path, 'r') as f:
        config = json.load(f)

    comfyui_address = config.get("comfyui_address")
    client_id = uuid.uuid4()
    with comfy_ws(comfyui_address, client_id) as ws:
        image = get_bboxes(image_bytes, ws, client_id, comfyui_address, description)
        
    image = Image.fromarray(image)
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="JPEG")
    
    return output_buffer.getvalue()