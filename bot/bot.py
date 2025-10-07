import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from io import BytesIO
import binascii

BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
API_URL = "http://127.0.0.1:8000/process/"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer("Привет! Отправь мне изображение, и я обработаю его 🚀")

@dp.message()
async def handle_image(message: types.Message):
    if not message.photo:
        await message.answer("Пожалуйста, отправь изображение 📷")
        return

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file.file_path)

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("file", file_bytes, filename="image.jpg", content_type="image/jpeg")
        async with session.post(API_URL, data=form) as resp:
            data = await resp.json()
            result_bytes = binascii.unhexlify(data["result"])
            output_image = BytesIO(result_bytes)
            await message.answer_photo(photo=output_image, caption="Вот результат обработки ✅")
