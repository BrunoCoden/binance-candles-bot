# alerts.py
import os
import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

from velas import (
    compute_channels,
    SYMBOL_DISPLAY, API_SYMBOL,
    CHANNEL_INTERVAL, STREAM_INTERVAL,
    RB_MULTI, RB_INIT_BAR
)
from paginado_binance import fetch_klines_paginado
from gSupertrend import compute_supertrend, _align_channels_to_stream, _has_data

ALERT_STREAM_BARS = int(os.getenv("ALERT_STREAM_BARS", "600"))
ALERT_CHANNEL_BARS = int(os.getenv("ALERT_CHANNEL_BARS", "300"))
ALERT_TOUCH_TOL = float(os.getenv("ALERT_TOUCH_TOL", "0.0"))
SUPER_ATR_PERIOD = int(os.getenv("SUPER_ATR_PERIOD", "10"))
SUPER_FACTOR = float(os.getenv("SUPER_FACTOR", "3.0"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_chat_ids_raw = os.getenv("TELEGRAM_CHAT_IDS", "")
TELEGRAM_CHAT_IDS = [
    part.strip()
    for part in _chat_ids_raw.replace(";", ",").split(",")
    if part.strip()
]
LOCAL_TZ_NAME = os.getenv("TZ", "UTC")
try:
    LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
except Exception:
    LOCAL_TZ = ZoneInfo("UTC")


def _prepare_frames():
    same_tf = STREAM_INTERVAL == CHANNEL_INTERVAL

    df_stream = fetch_klines_paginado(API_SYMBOL, STREAM_INTERVAL, ALERT_STREAM_BARS)
    if df_stream.empty:
        return None

    ohlc_stream = df_stream[["Open", "High", "Low", "Close", "Volume"]]

    if same_tf:
        df_channel = df_stream
    else:
        df_channel = fetch_klines_paginado(API_SYMBOL, CHANNEL_INTERVAL, ALERT_CHANNEL_BARS)
        if df_channel.empty:
            return None

    ohlc_channel = df_channel[["Open", "High", "Low", "Close", "Volume"]]

    channels = compute_channels(ohlc_channel, multi=RB_MULTI, init_bar=RB_INIT_BAR)

    if same_tf:
        chans_plot = channels.reindex(ohlc_stream.index).ffill()
    else:
        chans_plot = _align_channels_to_stream(channels, ohlc_stream.index)

    st_channel = compute_supertrend(ohlc_channel, atr_period=SUPER_ATR_PERIOD, factor=SUPER_FACTOR)

    if same_tf:
        st_aligned = st_channel.reindex(ohlc_stream.index).ffill()
    else:
        idx = ohlc_stream.index.union(st_channel.index)
        st_aligned = st_channel.reindex(idx).sort_index().ffill().reindex(ohlc_stream.index)

    return {
        "stream": ohlc_stream,
        "channels": chans_plot,
        "supertrend": st_aligned
    }


def _supertrend_alert(st_aligned: pd.DataFrame):
    if st_aligned is None or st_aligned.empty:
        return None

    direction = st_aligned.get("direction")
    supertrend_line = st_aligned.get("supertrend")
    if direction is None or supertrend_line is None:
        return None

    dir_series = direction.astype("float64")
    if dir_series.empty or dir_series.isna().all():
        return None

    last_idx = dir_series.index[-1]
    prev = dir_series.shift(1)
    if pd.isna(dir_series.iloc[-1]) or pd.isna(prev.iloc[-1]):
        return None

    if dir_series.iloc[-1] == prev.iloc[-1]:
        return None

    trend = "alcista" if dir_series.iloc[-1] < 0 else "bajista"
    price = supertrend_line.iloc[-1]

    return {
        "type": "supertrend_change",
        "timestamp": last_idx,
        "message": f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: Supertrend {trend} en {price:.2f}"
    }


def _touch_alerts(ohlc_stream: pd.DataFrame, channels: pd.DataFrame):
    if ohlc_stream is None or ohlc_stream.empty:
        return []

    latest = ohlc_stream.iloc[-1]
    idx = ohlc_stream.index[-1]

    val_upper = channels.get("ValueUpper")
    val_lower = channels.get("ValueLower")

    alerts = []

    if _has_data(val_upper):
        upper_level = float(val_upper.iloc[-1])
        if not np.isnan(upper_level):
            touched = _touched_level(latest, upper_level)
            if touched:
                alerts.append({
                    "type": "value_upper_touch",
                    "timestamp": idx,
                    "message": f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: Toque en ValueUpper {upper_level:.2f}"
                })

    if _has_data(val_lower):
        lower_level = float(val_lower.iloc[-1])
        if not np.isnan(lower_level):
            touched = _touched_level(latest, lower_level)
            if touched:
                alerts.append({
                    "type": "value_lower_touch",
                    "timestamp": idx,
                    "message": f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: Toque en ValueLower {lower_level:.2f}"
                })

    return alerts


def _touched_level(bar: pd.Series, level: float) -> bool:
    high = float(bar["High"])
    low = float(bar["Low"])
    if ALERT_TOUCH_TOL > 0:
        tol = max(level * ALERT_TOUCH_TOL, 1e-8)
        return (low - tol) <= level <= (high + tol)
    return low <= level <= high


def generate_alerts():
    frames = _prepare_frames()
    if not frames:
        return []

    alerts = []

    st_alert = _supertrend_alert(frames["supertrend"])
    if st_alert:
        alerts.append(st_alert)

    alerts.extend(_touch_alerts(frames["stream"], frames["channels"]))

    return alerts


def format_alert_message(alert: dict) -> str:
    ts = alert.get("timestamp")
    ts_str = ""

    if isinstance(ts, pd.Timestamp):
        try:
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts_local = ts.tz_convert(LOCAL_TZ)
            ts_str = ts_local.isoformat()
        except Exception:
            ts_str = str(ts)
    elif hasattr(ts, "astimezone"):
        try:
            ts_local = ts.astimezone(LOCAL_TZ)
            ts_str = ts_local.isoformat()
        except Exception:
            ts_str = str(ts)
    elif hasattr(ts, "isoformat"):
        ts_str = ts.isoformat()
    else:
        ts_str = str(ts)

    return f"{ts_str}\n{alert.get('message', '')}"


def send_alerts(alerts: list[dict]) -> int:
    if not alerts:
        return 0
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        return 0

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    sent = 0

    for alert in alerts:
        text = format_alert_message(alert)
        for chat_id in TELEGRAM_CHAT_IDS:
            payload = {
                "chat_id": chat_id,
                "text": text,
            }
            try:
                resp = requests.post(base_url, json=payload, timeout=10)
                resp.raise_for_status()
                sent += 1
            except Exception as exc:
                details = ""
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    try:
                        details = f" | Response: {exc.response.json()}"
                    except ValueError:
                        details = f" | Response: {exc.response.text}"
                print(f"[ERROR] Telegram send failed ({chat_id}): {exc}{details}")

    return sent


if __name__ == "__main__":
    alerts = generate_alerts()
    for alert in alerts:
        print(f"[ALERTA] {format_alert_message(alert)}")

    test_alert = {
        "type": "test_notification",
        "timestamp": pd.Timestamp.now(tz=LOCAL_TZ),
        "message": f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: alerta de prueba de Telegram"
    }
    print(f"[ALERTA][PRUEBA] {format_alert_message(test_alert)}")

    outgoing_alerts = alerts + [test_alert]

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS:
        sent = send_alerts(outgoing_alerts)
        print(f"[INFO] Alertas enviadas a Telegram: {sent}")
    else:
        print("[WARN] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_IDS no configurados; no se enviaron mensajes.")
