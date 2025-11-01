import os
import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

from paginado_binance import fetch_klines_paginado, INTERVAL_MS
from tabla_alertas import log_stream_bar
from backtest.realtime_backtest import process_realtime_signal, evaluate_realtime_risk
from velas import (
    SYMBOL_DISPLAY,
    API_SYMBOL,
    STREAM_INTERVAL,
    BB_DIRECTION,
    BB_LENGTH,
    BB_MULT,
    compute_bollinger_bands,
)


ALERT_STREAM_BARS = int(os.getenv("ALERT_STREAM_BARS", "5000"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_chat_ids_raw = os.getenv("TELEGRAM_CHAT_IDS", "")
TELEGRAM_CHAT_IDS = [part.strip() for part in _chat_ids_raw.replace(";", ",").split(",") if part.strip()]
SIGNAL_ALERTS_ENABLED = os.getenv("ALERT_ENABLE_BOLLINGER_SIGNALS", "false").lower() == "true"

LOCAL_TZ_NAME = os.getenv("TZ", "UTC")
try:
    LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
except Exception:
    LOCAL_TZ = ZoneInfo("UTC")


def _prepare_frames() -> dict | None:
    df_stream = fetch_klines_paginado(API_SYMBOL, STREAM_INTERVAL, ALERT_STREAM_BARS)
    if df_stream.empty:
        return None

    ohlc_stream = df_stream[["Open", "High", "Low", "Close", "Volume"]].copy()
    if "CloseTimeDT" in df_stream.columns:
        ohlc_stream["BarCloseTime"] = df_stream["CloseTimeDT"]
    else:
        interval_ms = INTERVAL_MS.get(STREAM_INTERVAL, 0)
        ohlc_stream["BarCloseTime"] = df_stream.index + pd.to_timedelta(interval_ms, unit="ms")
    bb = compute_bollinger_bands(ohlc_stream, BB_LENGTH, BB_MULT)
    bb_aligned = bb.reindex(ohlc_stream.index).ffill()

    return {
        "stream": ohlc_stream,
        "bollinger": bb_aligned,
    }


def _bollinger_alert(bb_aligned: pd.DataFrame, ohlc_stream: pd.DataFrame):
    if bb_aligned is None or bb_aligned.empty or ohlc_stream.empty:
        return None

    close = ohlc_stream["Close"].astype("float64")
    upper = bb_aligned.get("upper")
    lower = bb_aligned.get("lower")
    basis = bb_aligned.get("basis")

    if upper is None or lower is None or close.empty:
        return None

    if len(close) < 2 or len(upper) < 2 or len(lower) < 2:
        return None

    last_idx = close.index[-1]
    close_now = float(close.iloc[-1])
    close_prev = float(close.iloc[-2])
    upper_now = float(upper.iloc[-1])
    upper_prev = float(upper.iloc[-2])
    lower_now = float(lower.iloc[-1])
    lower_prev = float(lower.iloc[-2])

    if any(np.isnan(val) for val in (close_now, close_prev, upper_now, upper_prev, lower_now, lower_prev)):
        return None

    crossed_lower = close_prev <= lower_prev and close_now > lower_now
    crossed_upper = close_prev >= upper_prev and close_now < upper_now

    direction_filter = BB_DIRECTION

    if crossed_lower and direction_filter != -1:
        trend = "alcista"
        direction = "long"
        ref_price = lower_now
        trigger_price = lower_now
    elif crossed_upper and direction_filter != 1:
        trend = "bajista"
        direction = "short"
        ref_price = upper_now
        trigger_price = upper_now
    else:
        return None

    last_bar = ohlc_stream.iloc[-1]
    bar_close_ts = last_bar.get("BarCloseTime", last_idx)
    volume = float(last_bar.get("Volume", np.nan))
    basis_now = float(basis.iloc[-1]) if basis is not None else np.nan

    return {
        "type": "bollinger_signal",
        "timestamp": bar_close_ts if isinstance(bar_close_ts, pd.Timestamp) else last_idx,
        "message": (
            f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: Señal Bollinger {trend} en {trigger_price:.2f} "
            f"(banda de referencia {ref_price:.2f})"
        ),
        "price": trigger_price,
        "close_price": close_now,
        "direction": direction,
        "basis": basis_now,
        "reference_band": ref_price,
        "volume": volume,
    }



def generate_alerts() -> list[dict]:
    frames = _prepare_frames()
    if not frames:
        return []

    log_stream_bar(frames["stream"])
    try:
        evaluate_realtime_risk(frames["stream"], profile="tr")
    except Exception as exc:
        print(f"[ALERT][WARN] No se pudo evaluar SL/TP en tiempo real ({exc})")
    if not SIGNAL_ALERTS_ENABLED:
        return []

    alert = _bollinger_alert(frames["bollinger"], frames["stream"])
    if alert:
        try:
            process_realtime_signal(alert, profile="tr")
        except Exception as exc:
            print(f"[ALERT][WARN] No se pudo actualizar el backtest en tiempo real ({exc})")
        return [alert]
    return []


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

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS:
        sent = send_alerts(alerts)
        print(f"[INFO] Alertas enviadas a Telegram: {sent}")
    else:
        print("[WARN] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_IDS no configurados; no se enviaron mensajes.")
