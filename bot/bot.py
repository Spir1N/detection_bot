import aiohttp, os
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import BufferedInputFile, InlineKeyboardMarkup, InlineKeyboardButton
import binascii

with open('bot/.secret', 'r') as fh:
    vars_dict = dict(
        tuple(line.replace('\n', '').split('='))
        for line in fh.readlines() if not line.startswith('#')
    )

os.environ.update(vars_dict)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = "http://127.0.0.1:8000/process/"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
user_state = {}

model_kb = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="YOLO", callback_data="model_yolo")],
    [InlineKeyboardButton(text="GroundingDINO", callback_data="model_gd")],
    [InlineKeyboardButton(text="RCNN (Beta)", callback_data="model_rcnn")],
    [InlineKeyboardButton(text="DETR", callback_data="model_detr")],
])


@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer("Привет! Отправь мне изображение, и я обработаю его\n\nЕсли введешь команду /model, то сможешь выбрать нужную модель для детекции.")


@dp.message(Command("model"))
async def choose_model(message: types.Message):
    await message.answer("Выбери модель для детекции:", reply_markup=model_kb)


@dp.callback_query(F.data.startswith("model_"))
async def model_selected(callback: types.CallbackQuery):
    model = callback.data.split("_")[1]
    user_id = callback.from_user.id

    user_state[user_id] = {"model": model, "await_desc": False, "description": ""}

    if model == "gd":
        user_state[user_id]["await_desc"] = True
        await callback.message.answer("Введите описание объекта (до 255 символов):")
    else:
        await callback.message.answer(f"Модель установлена: {model.upper()}")

    await callback.answer()


@dp.message()
async def handle_all(message: types.Message):
    user_id = message.from_user.id
    state = user_state.get(user_id)

    if state and state.get("await_desc"):
        desc = message.text.strip()

        if len(desc) > 255:
            await message.answer("Описание слишком длинное. Максимум 255 символов")
            return

        user_state[user_id]["description"] = desc
        user_state[user_id]["await_desc"] = False
        await message.answer(f"Описание установлено. Теперь отправьте изображение")
        return

    if not message.photo:
        await message.answer("Пожалуйста, отправь изображение или введи /model для смены модели детекции.")
        return

    model = state["model"] if state else "yolo"
    description = state.get("description") if state else ""

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file.file_path)

    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("file", file_bytes, filename="image.jpg", content_type="image/jpeg")
        form.add_field("model", model)
        form.add_field("description", description)

        async with session.post(API_URL, data=form) as resp:
            data = await resp.json()

            result_bytes = binascii.unhexlify(data["result"])
            output_file = BufferedInputFile(result_bytes, filename="processed.jpg")
            await message.answer_photo(photo=output_file,
                                       caption=f"Готово! (Модель: {model.upper()})")