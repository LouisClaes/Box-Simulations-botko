import asyncio
import os

# Load .env
from pathlib import Path
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

from src.monitoring.telegram_notifier import send_telegram

async def test():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    
    print(f"Token: {token[:20]}... (length: {len(token)})")
    print(f"Chat ID: {chat_id}")
    
    result = await send_telegram("🤖 Botko overnight runner is ready!\n\nThis is a test notification from Box-Simulations-botko.")
    
    if result:
        print("✅ Telegram notification sent successfully!")
    else:
        print("❌ Failed to send notification")
    
    return result

if __name__ == "__main__":
    asyncio.run(test())
