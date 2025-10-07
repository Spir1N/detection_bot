import io
from PIL import Image, ImageDraw

# Пример — простая заглушка вместо твоей модели
# В твоём коде тут можно подключить torch.load(), model.predict() и т.д.
def detect_objects(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    # Пример: просто рисуем рамку (вместо реальной детекции)
    draw.rectangle((50, 50, 200, 200), outline="red", width=5)

    output_buffer = io.BytesIO()
    image.save(output_buffer, format="JPEG")
    return output_buffer.getvalue()
