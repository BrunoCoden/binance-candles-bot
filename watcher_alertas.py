# watcher_alertas.py
import os
import time
from datetime import datetime, timezone

import pandas as pd

from alerts import generate_alerts, send_alerts, format_alert_message
from trade_logger import send_trade_notification, format_timestamp
from velas import SYMBOL_DISPLAY, STREAM_INTERVAL

POLL_SECONDS = float(os.getenv("ALERT_POLL_SECONDS", "5"))
MAX_SEEN = int(os.getenv("ALERT_MAX_SEEN", "500"))
SEND_STARTUP_TEST = os.getenv("WATCHER_STARTUP_TEST_ALERT", "true").lower() == "true"


def _notify_startup():
    if not SEND_STARTUP_TEST:
        return
    try:
        now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
        timestamp = format_timestamp(now_utc)
        message = (
            f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}\n"
            f"[PRUEBA] Watcher iniciado\n"
            f"Hora: {timestamp}\n"
            f"Este es un mensaje de verificación del formato de alertas."
        )
        send_trade_notification(message)
    except Exception as exc:
        print(f"[WATCHER][WARN] No se pudo enviar la alerta de prueba: {exc}")


def main():
    seen = []
    _notify_startup()
    while True:
        try:
            events = generate_alerts()
        except Exception as exc:
            print(f"[ERROR] {exc}")
            time.sleep(POLL_SECONDS)
            continue

        new_alerts = []
        for evt in events:
            ts = evt.get("timestamp")
            key = (evt.get("type"), ts)
            if key in seen:
                continue
            seen.append(key)
            seen[:] = seen[-MAX_SEEN:]

            print(f"[ALERTA] {format_alert_message(evt)}")
            new_alerts.append(evt)

        if new_alerts:
            send_alerts(new_alerts)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
