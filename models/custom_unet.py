import io
from PIL import Image, ImageDraw

def detect_objects(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 200, 200), outline="red", width=5)

    output_buffer = io.BytesIO()
    image.save(output_buffer, format="JPEG")
    return output_buffer.getvalue()
