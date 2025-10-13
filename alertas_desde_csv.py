# alertas_desde_csv.py
# ---------------------------------------------------------
# Lee el CSV generado por velas/tu loop y dispara alertas cuando
# detecta nuevas velas cerradas con Buy=1 o Sell=1.
#
# Soporta:
#   - Consola (siempre)
#   - Beep (winsound) opcional
#   - Notificación Windows (win10toast) opcional
#   - Webhook genérico opcional
#   - Telegram:
#       * TELEGRAM_CHAT_ID: envía a un chat fijo (privado o grupo -100...)
#       * TELEGRAM_BROADCAST_ALL=1: difunde a todos los chats del targets.json
#         o aprendidos por getUpdates (guardados en telegram_targets.json)
#
# ENV (en alerts.env o .env):
#   CSV_PATH=tabla.csv
#   SYMBOL=ETHUSDT.P
#   ALERT_POLL_SEC=2
#   ENABLE_BEEP=1
#   ENABLE_TOAST=0
#   ALERT_WEBHOOK_URL=https://...
#   TELEGRAM_BOT_TOKEN=123456789:AA...
#   TELEGRAM_CHAT_ID=-1001234567890   (opcional)
#   TELEGRAM_BROADCAST_ALL=1
#   TELEGRAM_TARGETS_PATH=telegram_targets.json
#   TELEGRAM_REFRESH_UPDATES_SEC=60
#   ALERT_TEST=1           -> dispara alerta de prueba al inicio
# ---------------------------------------------------------

import os
import time
import json
import platform
from typing import Optional, List
import pandas as pd

# ===== Carga de entorno (alerts.env o fallback .env) =====
try:
    from dotenv import load_dotenv
    if os.path.exists("alerts.env"):
        load_dotenv("alerts.env")
    else:
        if os.path.exists(".env"):
            load_dotenv(".env")
except Exception:
    pass

# ===== Config =====
CSV_PATH        = os.getenv("CSV_PATH", "stream_table.csv").strip()
SYMBOL          = os.getenv("SYMBOL", "ETHUSDT.P").strip()
POLL_SEC        = int(os.getenv("ALERT_POLL_SEC", "2"))

ENABLE_BEEP     = os.getenv("ENABLE_BEEP", "1") == "1"
ENABLE_TOAST    = os.getenv("ENABLE_TOAST", "0") == "1"

WEBHOOK_URL     = os.getenv("ALERT_WEBHOOK_URL", "").strip()

TG_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TG_BROADCAST    = os.getenv("TELEGRAM_BROADCAST_ALL", "0") == "1"
TG_TARGETS_PATH = os.getenv("TELEGRAM_TARGETS_PATH", "telegram_targets.json").strip()
TG_REFRESH_SEC  = int(os.getenv("TELEGRAM_REFRESH_UPDATES_SEC", "60"))

ALERT_TEST      = os.getenv("ALERT_TEST", "0") == "1"

# Dependencia opcional para red
try:
    import requests
except Exception:
    requests = None

# ---------- Utilidades locales ----------
def _beep():
    if not ENABLE_BEEP:
        return
    try:
        if platform.system().lower().startswith("win"):
            import winsound
            winsound.Beep(880, 200)
            winsound.Beep(660, 150)
        else:
            # campanita estándar en *nix
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
        pass

def _post_webhook(payload: dict):
    if not WEBHOOK_URL or not requests:
        return
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception:
        pass

def _fmt_row(r) -> str:
    return (f"[{r.get('Date','')}] O:{r.get('Open','')} H:{r.get('High','')} "
            f"L:{r.get('Low','')} C:{r.get('Close','')} Vol:{r.get('Volume','')}")

# ---------- Telegram helpers ----------
class TelegramTargets:
    """
    Soporta dos formatos de archivo:
      A) dict:
         {
           "last_update_id": 0,
           "targets": {
             "<chat_id>": {"type": "...", "title": "..."},
             ...
           }
         }
      B) lista:
         [
           {"chat_id": -100..., "name": "grupo", "type": "supergroup"},
           {"chat_id": 123...,   "title": "privado", "type": "private"}
         ]
    """
    def __init__(self, path: str):
        self.path = path
        self.data = {"last_update_id": 0, "targets": {}}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            # inicial vacío
            self.data = {"last_update_id": 0, "targets": {}}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Formato A (dict con "targets")
            if isinstance(raw, dict) and "targets" in raw:
                self.data = raw
                self.data.setdefault("last_update_id", 0)
                self.data.setdefault("targets", {})
            # Formato B (lista de objetos con chat_id)
            elif isinstance(raw, list):
                self.data = {"last_update_id": 0, "targets": {}}
                for item in raw:
                    cid = item.get("chat_id")
                    if cid is None:
                        continue
                    cid = str(cid)
                    title = item.get("name") or item.get("title") or ""
                    ttype = item.get("type", "")
                    if cid not in self.data["targets"]:
                        self.data["targets"][cid] = {"type": ttype, "title": title}
            else:
                self.data = {"last_update_id": 0, "targets": {}}
        except Exception:
            self.data = {"last_update_id": 0, "targets": {}}

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            pass

    @property
    def last_update_id(self) -> int:
        return int(self.data.get("last_update_id", 0))

    @last_update_id.setter
    def last_update_id(self, v: int):
        self.data["last_update_id"] = int(v)

    def add_chat(self, chat_id: int | str, chat_type: str, title: str):
        chat_id = str(chat_id)
        if chat_id not in self.data["targets"]:
            self.data["targets"][chat_id] = {"type": chat_type or "", "title": title or ""}
        else:
            # actualizar metadatos
            if chat_type:
                self.data["targets"][chat_id]["type"] = chat_type
            if title:
                self.data["targets"][chat_id]["title"] = title

    def list_chat_ids(self) -> List[str]:
        return list(self.data.get("targets", {}).keys())

def tg_api(method: str, payload: dict):
    if not TG_TOKEN or not requests:
        return None
    url = f"https://api.telegram.org/bot{TG_TOKEN}/{method}"
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"[WARN] Telegram HTTP {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[WARN] Telegram error: {e}")
    return None

def tg_send_text(chat_id: str | int, text: str):
    return tg_api("sendMessage", {"chat_id": chat_id, "text": text})

def tg_collect_updates(store: TelegramTargets):
    # Levanta updates para aprender chats nuevos y persistirlos.
    offset = store.last_update_id + 1 if store.last_update_id else None
    payload = {"timeout": 0, "allowed_updates": ["message", "my_chat_member", "chat_member", "channel_post"]}
    if offset:
        payload["offset"] = offset

    resp = tg_api("getUpdates", payload)
    if not resp or not resp.get("ok", False):
        return
    max_update_id = store.last_update_id
    for upd in resp.get("result", []):
        max_update_id = max(max_update_id, int(upd.get("update_id", 0)))
        for field in ["message", "channel_post", "my_chat_member", "chat_member"]:
            msg = upd.get(field)
            if msg and "chat" in msg:
                chat = msg["chat"]
                chat_id = chat.get("id")
                chat_type = chat.get("type")
                title = chat.get("title") or chat.get("username") or chat.get("first_name") or ""
                if chat_id is not None:
                    store.add_chat(chat_id, chat_type, title)
    if max_update_id > store.last_update_id:
        store.last_update_id = max_update_id
        store.save()

# ---------- Notificación ----------
def _notify_all(signal: str, row: pd.Series):
    text = f"{SYMBOL} — {signal}\n{_fmt_row(row)}"
    print(f"[ALERTA] {signal} | {_fmt_row(row)}")
    _beep()
    _toast(f"{SYMBOL} — {signal}", _fmt_row(row))
    _post_webhook({"symbol": SYMBOL, "signal": signal, "data": dict(row)})

    if not (TG_TOKEN and requests):
        return

    # Chat fijo (si está)
    if TG_CHAT_ID:
        try:
            tg_send_text(TG_CHAT_ID, text)
        except Exception:
            pass

    # Broadcast (si está activo)
    if TG_BROADCAST:
        store = TelegramTargets(TG_TARGETS_PATH)
        try:
            tg_collect_updates(store)  # aprende chats si hay activity
        except Exception:
            pass
        for cid in store.list_chat_ids():
            # evita duplicar si TG_CHAT_ID coincide
            if TG_CHAT_ID and str(cid) == str(TG_CHAT_ID):
                continue
            try:
                tg_send_text(cid, text)
            except Exception:
                continue

# ---------- CSV ----------
def _load_df_safe() -> Optional[pd.DataFrame]:
    if not os.path.exists(CSV_PATH):
        return None
    try:
        return pd.read_csv(CSV_PATH)
    except Exception:
        return None

# ---------- Loop ----------
def run_alerts():
    print(f"[INFO] Alertas CSV: {CSV_PATH} | SYMBOL={SYMBOL} | poll={POLL_SEC}s")
    if TG_TOKEN:
        print(f"[INFO] Telegram bot: ON | Broadcast={'ON' if TG_BROADCAST else 'OFF'} | targets={TG_TARGETS_PATH}")
    else:
        print(f"[WARN] Telegram bot: OFF (sin TELEGRAM_BOT_TOKEN)")

    if ALERT_TEST:
        dummy = pd.Series({"Date": "TEST", "Open": 0, "High": 0, "Low": 0, "Close": 0, "Volume": 0})
        _notify_all("🚀 TEST ALERT", dummy)

    last_key = None
    df0 = _load_df_safe()
    if df0 is not None and not df0.empty:
        key_col0 = "CloseTimeMs" if "CloseTimeMs" in df0.columns else "Date"
        last_key = df0[key_col0].iloc[-1]

    last_refresh = 0.0
    while True:
        try:
            df = _load_df_safe()
            if df is None or df.empty:
                time.sleep(POLL_SEC)
                continue

            key_col = "CloseTimeMs" if "CloseTimeMs" in df.columns else "Date"
            df = df.sort_values(key_col)

            # detectar nuevas filas
            if last_key is None:
                new_rows = df.tail(1)
            else:
                try:
                    if key_col == "CloseTimeMs":
                        df[key_col] = pd.to_numeric(df[key_col], errors="coerce")
                        lk_num = pd.to_numeric(pd.Series([last_key]), errors="coerce").iloc[0]
                        new_rows = df[df[key_col] > lk_num]
                    else:
                        pos = df.index[df[key_col] == last_key]
                        new_rows = df.loc[pos[0]+1:] if len(pos) else df.tail(1)
                except Exception:
                    new_rows = df.tail(1)

            # procesar nuevas señales
            for _, r in new_rows.iterrows():
                buy  = int(r.get("Buy", 0))  if pd.notna(r.get("Buy", None))  else 0
                sell = int(r.get("Sell", 0)) if pd.notna(r.get("Sell", None)) else 0

                if buy == 1 and sell != 1:
                    _notify_all("▲ BUY", r)
                elif sell == 1 and buy != 1:
                    _notify_all("▼ SELL", r)
                elif buy == 1 and sell == 1:
                    _notify_all("⚠ Señales simultáneas (BUY & SELL)", r)

                last_key = r.get(key_col, last_key)

            # refrescar updates cada TG_REFRESH_SEC para aprender chats
            now = time.time()
            if TG_TOKEN and TG_BROADCAST and requests and (now - last_refresh >= TG_REFRESH_SEC):
                try:
                    store = TelegramTargets(TG_TARGETS_PATH)
                    tg_collect_updates(store)
                except Exception:
                    pass
                last_refresh = now

            time.sleep(POLL_SEC)

        except KeyboardInterrupt:
            print("\n[EXIT] Cortado por usuario.")
            break
        except Exception as e:
            print(f"[WARN] {type(e).__name__}: {e}")
            time.sleep(POLL_SEC)

# ---------- Main ----------
if __name__ == "__main__":
    run_alerts()
