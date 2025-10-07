import asyncio
import uvicorn
from api.server import app
from bot.bot import dp, bot

async def start_bot():
    await dp.start_polling(bot)

async def start_api():
    uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(uvicorn_config)
    await server.serve()

async def main():
    await asyncio.gather(start_api(), start_bot())

if __name__ == "__main__":
    asyncio.run(main())
