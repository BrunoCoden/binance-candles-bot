# watcher_alertas.py
import os
import time

from alerts import generate_alerts, send_alerts, format_alert_message

POLL_SECONDS = float(os.getenv("ALERT_POLL_SECONDS", "5"))
MAX_SEEN = int(os.getenv("ALERT_MAX_SEEN", "500"))


def main():
    seen = []
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
