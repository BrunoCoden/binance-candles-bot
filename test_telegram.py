import os
import requests

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "TU_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "TU_CHAT_ID")

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": "🚨 Alerta de prueba desde Python",
    "parse_mode": "HTML"
}

r = requests.post(url, json=payload)
print(r.status_code, r.text)
