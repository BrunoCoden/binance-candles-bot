# alertas_desde_csv.py
# ---------------------------------------------------------
# Lee el stream CSV generado por velas_tv_sin_sdk.py y dispara alertas
# cuando detecta nuevas velas cerradas con Buy=1 o Sell=1.
#
# - No requiere Binance (lee el CSV).
# - Evita duplicados usando CloseTimeMs si existe; si no, usa Date.
# - Soporta:
#     * Consola (siempre)
#     * Beep en Windows (winsound) opcional
#     * Notificación de escritorio en Windows (win10toast) opcional
#     * Webhook genérico (ALERT_WEBHOOK_URL) opcional
#     * Telegram (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID) opcional
#
# ENV útiles:
#   CSV_PATH=stream_table.csv
#   SYMBOL=ETHUSDT.P
#   ALERT_POLL_SEC=2
#   ENABLE_BEEP=1
#   ENABLE_TOAST=1
#   ALERT_WEBHOOK_URL=https://tu.webhook
#   TELEGRAM_BOT_TOKEN=...
#   TELEGRAM_CHAT_ID=...
# ---------------------------------------------------------

import os
import time
import json
import platform
import pandas as pd

CSV_PATH        = os.getenv("CSV_PATH", "stream_table.csv")
SYMBOL          = os.getenv("SYMBOL", "ETHUSDT.P")
POLL_SEC        = int(os.getenv("ALERT_POLL_SEC", "2"))

ENABLE_BEEP     = os.getenv("ENABLE_BEEP", "1") == "1"
ENABLE_TOAST    = os.getenv("ENABLE_TOAST", "0") == "1"  # por defecto OFF para evitar deps

WEBHOOK_URL     = os.getenv("ALERT_WEBHOOK_URL", "").strip()
TG_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Opcionales
try:
    import requests
except Exception:
    requests = None

def _beep():
    if not ENABLE_BEEP:
        return
    try:
        if platform.system().lower().startswith("win"):
            import winsound
            winsound.Beep(880, 200)  # 880Hz 200ms
            winsound.Beep(660, 150)
        else:
            # Unix-like: campana
            print("\a", end="", flush=True)
    except Exception:
        pass

def _toast(title: str, msg: str):
    if not ENABLE_TOAST:
        return
    try:
        from win10toast import ToastNotifier
        ToastNotifier().show_toast(title, msg, duration=4, threaded=True)
    except Exception:
        # Silencioso si no está win10toast instalado
        pass

def _post_webhook(payload: dict):
    if not WEBHOOK_URL or not requests:
        return
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception:
        pass

def _send_telegram(text: str):
    if not (TG_TOKEN and TG_CHAT_ID and requests):
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text}, timeout=5)
    except Exception:
        pass

def _fmt_row(r) -> str:
    # r es una Serie de pandas (última vela)
    return (f"[{r.get('Date','')}] O:{r.get('Open','')} H:{r.get('High','')} "
            f"L:{r.get('Low','')} C:{r.get('Close','')} Vol:{r.get('Volume','')}")

def _notify(signal: str, row: pd.Series):
    txt = f"{SYMBOL} — {signal}\n{_fmt_row(row)}"
    print(f"[ALERTA] {signal} | {_fmt_row(row)}")
    _beep()
    _toast(f"{SYMBOL} — {signal}", _fmt_row(row))
    _post_webhook({"symbol": SYMBOL, "signal": signal, "data": dict(row)})
    _send_telegram(f"{SYMBOL} — {signal}\n{_fmt_row(row)}")

def _load_df_safe() -> pd.DataFrame | None:
    if not os.path.exists(CSV_PATH):
        return None
    try:
        # Intento de lectura tolerante a archivos en escritura
        return pd.read_csv(CSV_PATH)
    except Exception:
        return None

def run_alerts():
    print(f"[INFO] Alertas leyendo CSV: {CSV_PATH} | SYMBOL={SYMBOL} | poll={POLL_SEC}s")
    last_key = None  # CloseTimeMs si existe; si no, Date (string)

    # Inicializar last_key con la última fila del CSV (si existe)
    df0 = _load_df_safe()
    if df0 is not None and not df0.empty:
        key_col = "CloseTimeMs" if "CloseTimeMs" in df0.columns else "Date"
        last_key = df0[key_col].iloc[-1]

    while True:
        try:
            df = _load_df_safe()
            if df is None or df.empty:
                time.sleep(POLL_SEC)
                continue

            # Determinar clave para deduplicar
            key_col = "CloseTimeMs" if "CloseTimeMs" in df.columns else "Date"
            # Orden lógico por la clave
            df = df.sort_values(key_col)

            # Filtrar nuevas filas
            if last_key is None:
                new_rows = df.tail(1)  # primera pasada: sólo la última
            else:
                if key_col not in df.columns:
                    new_rows = pd.DataFrame()  # algo raro; no alertamos
                else:
                    # Si clave es numérica comparamos numéricamente
                    try:
                        if key_col == "CloseTimeMs":
                            df[key_col] = pd.to_numeric(df[key_col], errors="coerce")
                            lk_num = pd.to_numeric(pd.Series([last_key]), errors="coerce").iloc[0]
                            new_rows = df[df[key_col] > lk_num]
                        else:
                            # Para Date (string), comparamos por posición posterior
                            pos = df.index[df[key_col] == last_key]
                            if len(pos) == 0:
                                new_rows = df.tail(1)
                            else:
                                new_rows = df.loc[pos[0]+1:]
                    except Exception:
                        new_rows = df.tail(1)

            # Procesar nuevas filas en orden
            for _, r in new_rows.iterrows():
                # Normalizar campos esperados
                buy  = int(r.get("Buy", 0)) if pd.notna(r.get("Buy", None)) else 0
                sell = int(r.get("Sell", 0)) if pd.notna(r.get("Sell", None)) else 0

                if buy == 1 and sell != 1:
                    _notify("▲ BUY", r)
                elif sell == 1 and buy != 1:
                    _notify("▼ SELL", r)
                elif buy == 1 and sell == 1:
                    _notify("⚠ Señales simultáneas (BUY & SELL)", r)

                last_key = r.get(key_col, last_key)

            time.sleep(POLL_SEC)

        except KeyboardInterrupt:
            print("\n[EXIT] Cortado por usuario.")
            break
        except Exception as e:
            print(f"[WARN] {type(e).__name__}: {e}")
            time.sleep(POLL_SEC)

if __name__ == "__main__":
    run_alerts()
