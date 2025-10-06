# alertaSinSenal.py
# ---------------------------------------------------------
# Alertas en TIEMPO REAL (intrabar) basadas en soloAlertas.compute_channels:
# Dispara cuando el precio toca UpperQ o LowerQ en la vela en curso.
#
# Salidas soportadas:
#   - Consola (siempre)
#   - Beep (opcional)
#   - Notificación Windows (opcional)
#   - Webhook genérico (opcional)
#   - Telegram: chat fijo y/o broadcast a targets descubiertos
#
# ENV (alerts.env o variables de entorno):
#   SYMBOL=ETHUSDT.P
#   INTERVAL=30m
#   LIMIT=800
#   RB_MULTI=4.0
#   RB_INIT_BAR=301
#   ALERT_POLL_SEC=2
#   ENABLE_BEEP=1
#   ENABLE_TOAST=0
#   ALERT_WEBHOOK_URL=https://...
#   TELEGRAM_BOT_TOKEN=...
#   TELEGRAM_CHAT_ID=-1001234567890  (opcional)
#   TELEGRAM_BROADCAST_ALL=1
#   TELEGRAM_TARGETS_PATH=telegram_targets.json
#   TELEGRAM_REFRESH_UPDATES_SEC=60
#   ALERT_TEST=0
# ---------------------------------------------------------

import os
import time
import json
import platform
from typing import Optional, List, Tuple

import pandas as pd

# Cargar .env de alertas si existe
try:
    from dotenv import load_dotenv
    if os.path.exists("alerts.env"):
        load_dotenv("alerts.env")
except Exception:
    pass

# Importa lógica y config de soloAlertas
from soloAlertas import (
    fetch_klines, compute_channels,
    SYMBOL_DISPLAY, API_SYMBOL, INTERVAL, LIMIT,
    RB_MULTI, RB_INIT_BAR
)

# ================== Config alertas ==================
SYMBOL          = os.getenv("SYMBOL", SYMBOL_DISPLAY)
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

# Dependencia opcional de red
try:
    import requests
except Exception:
    requests = None

# ================== Utils locales ==================
def _fmt_row_ts(ts) -> str:
    try:
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)

def _fmt_row(ohlc_row: pd.Series) -> str:
    return (f"[{_fmt_row_ts(ohlc_row.name)}] "
            f"O:{ohlc_row.get('Open','')} H:{ohlc_row.get('High','')} "
            f"L:{ohlc_row.get('Low','')} C:{ohlc_row.get('Close','')} Vol:{ohlc_row.get('Volume','')}")

def _beep():
    if not ENABLE_BEEP:
        return
    try:
        if platform.system().lower().startswith("win"):
            import winsound
            winsound.Beep(880, 200)
            winsound.Beep(660, 150)
        else:
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

# ================== Telegram helpers ==================
class TelegramTargets:
    def __init__(self, path: str):
        self.path = path
        self.data = {"last_update_id": 0, "targets": {}}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                self.data.setdefault("last_update_id", 0)
                self.data.setdefault("targets", {})
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
            self.data["targets"][chat_id] = {"type": chat_type, "title": title or ""}
        else:
            self.data["targets"][chat_id]["type"] = chat_type
            if title:
                self.data["targets"][chat_id]["title"] = title

    def list_chat_ids(self) -> List[str]:
        return list(self.data.get("targets", {}).keys())

def tg_api(method: str, payload: dict):
    if not (TG_TOKEN and requests):
        return None
    url = f"https://api.telegram.org/bot{TG_TOKEN}/{method}"
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def tg_send_text(chat_id: str | int, text: str):
    tg_api("sendMessage", {"chat_id": chat_id, "text": text})

def tg_collect_updates(store: TelegramTargets):
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

# ================== Notificación unificada ==================
def _notify_all(signal: str, ohlc_row: pd.Series, where: str):
    text = f"{SYMBOL} — {signal} ({where})\n{_fmt_row(ohlc_row)}"
    print(f"[ALERTA-RT] {signal} [{where}] | {_fmt_row(ohlc_row)}")
    _beep()
    _toast(f"{SYMBOL} — {signal}", f"{where} | {_fmt_row(ohlc_row)}")
    _post_webhook({"symbol": SYMBOL, "signal": signal, "where": where, "data": dict(ohlc_row)})

    if not (TG_TOKEN and requests):
        return
    # Chat fijo
    if TG_CHAT_ID:
        try:
            tg_send_text(TG_CHAT_ID, text)
        except Exception:
            pass
    # Broadcast
    if TG_BROADCAST:
        store = TelegramTargets(TG_TARGETS_PATH)
        try:
            tg_collect_updates(store)
        except Exception:
            pass
        for cid in store.list_chat_ids():
            if TG_CHAT_ID and str(cid) == str(TG_CHAT_ID):
                continue
            try:
                tg_send_text(cid, text)
            except Exception:
                continue

# ================== Core: detección intrabar ==================
class RTDeDup:
    """Evita duplicados por barra y por lado (UpperQ/LowerQ)."""
    def __init__(self):
        self.last_bar_key: Optional[int] = None  # CloseTime de la vela en curso
        self.fired_upper: bool = False
        self.fired_lower: bool = False

    def should_fire(self, bar_key: int, side: str) -> bool:
        # Si cambia de vela, resetea flags
        if self.last_bar_key != bar_key:
            self.last_bar_key = bar_key
            self.fired_upper = False
            self.fired_lower = False
        if side == "UpperQ" and not self.fired_upper:
            self.fired_upper = True
            return True
        if side == "LowerQ" and not self.fired_lower:
            self.fired_lower = True
            return True
        return False

def _detect_intrabar_signals(df: pd.DataFrame, chans: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
    """
    Revisa SOLO la última vela (en curso) y retorna lista de señales [(label, row), ...]
    label ∈ {"TOUCH UpperQ", "TOUCH LowerQ"}
    """
    if df.empty or chans.empty:
        return []
    last_row = df.iloc[-1]
    last_idx = df.index[-1]

    # Si el índice es DatetimeIndex con tz, todo bien; si no, igual sirve para imprimir.
    try:
        # CloseTime viene como ms en df original, pero aquí usamos el índice de pandas
        pass
    except Exception:
        pass

    # Toques intrabar (last row no cerrada): Low <= Q <= High
    signals = []
    uq = chans.loc[last_idx, "UpperQ"] if pd.notna(chans.loc[last_idx, "UpperQ"]) else None
    lq = chans.loc[last_idx, "LowerQ"] if pd.notna(chans.loc[last_idx, "LowerQ"]) else None

    lo = float(last_row["Low"]); hi = float(last_row["High"])
    if uq is not None and lo <= float(uq) <= hi:
        signals.append(("TOUCH UpperQ", last_row))
    if lq is not None and lo <= float(lq) <= hi:
        signals.append(("TOUCH LowerQ", last_row))
    return signals

# ================== Loop principal ==================
def run_alerts_realtime():
    print(f"[INFO] Alertas RT: SYMBOL={SYMBOL} | INTERVAL={INTERVAL} | poll={POLL_SEC}s")
    if TG_TOKEN:
        print(f"[INFO] Telegram bot activo. Broadcast={'ON' if TG_BROADCAST else 'OFF'} | targets={TG_TARGETS_PATH}")
    if ALERT_TEST:
        dummy = pd.Series({"Open": 0, "High": 0, "Low": 0, "Close": 0, "Volume": 0})
        dummy.name = pd.Timestamp.utcnow()
        _notify_all("🚀 TEST ALERT", dummy, where="TEST")

    dedup = RTDeDup()
    last_refresh = 0.0

    while True:
        try:
            # 1) Bajamos velas con la vela en curso incluida
            df = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
            ohlc = df[["Open","High","Low","Close","Volume"]]

            # 2) Canales recalculados sobre TODO el tramo, incluyendo la vela actual
            chans = compute_channels(ohlc, multi=RB_MULTI, init_bar=RB_INIT_BAR)

            # 3) Detectar señales intrabar en la última vela
            sigs = _detect_intrabar_signals(ohlc, chans)

            # 4) Clave única por vela actual: usamos CloseTime (ms) de la última fila raw
            try:
                bar_key = int(df.iloc[-1]["CloseTime"])
            except Exception:
                # Fallback a timestamp del índice
                bar_key = int(pd.Timestamp(df.index[-1]).value // 10**6)  # ms

            for label, row in sigs:
                side = "UpperQ" if "Upper" in label else "LowerQ"
                if dedup.should_fire(bar_key, side):
                    _notify_all(label, row, where="REALTIME")

            # 5) Refrescar lista de targets de Telegram de vez en cuando
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

# ================== Main ==================
if __name__ == "__main__":
    run_alerts_realtime()
