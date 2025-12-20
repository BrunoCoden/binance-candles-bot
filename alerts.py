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
STOP_LOSS_PCT = float(os.getenv("STRAT_STOP_LOSS_PCT", "0.05"))
TAKE_PROFIT_PCT = float(os.getenv("STRAT_TAKE_PROFIT_PCT", "0.095"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_chat_ids_raw = os.getenv("TELEGRAM_CHAT_IDS", "")
TELEGRAM_CHAT_IDS = [part.strip() for part in _chat_ids_raw.replace(";", ",").split(",") if part.strip()]
SIGNAL_ALERTS_ENABLED = os.getenv("ALERT_ENABLE_BOLLINGER_SIGNALS", "false").lower() == "true"
_last_direction: str | None = None
_pending_break: dict | None = None  # Guarda rotura pendiente hasta que haya cierre de rebote

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

    close_series = ohlc_stream["Close"].astype("float64")
    upper = bb_aligned.get("upper")
    lower = bb_aligned.get("lower")
    basis = bb_aligned.get("basis")

    if upper is None or lower is None or close_series.empty:
        return None

    # Usa solo velas cerradas: descarta la última fila (vela en curso)
    if len(close_series) < 2 or len(upper) < 2 or len(lower) < 2:
        return None
    closed_close = close_series.iloc[:-1]
    closed_upper = upper.iloc[:-1]
    closed_lower = lower.iloc[:-1]
    closed_basis = basis.iloc[:-1] if basis is not None else None

    if closed_close.empty:
        return None

    last_idx = closed_close.index[-1]
    close_now = float(closed_close.iloc[-1])
    upper_now = float(closed_upper.iloc[-1])
    lower_now = float(closed_lower.iloc[-1])

    if any(np.isnan(val) for val in (close_now, upper_now, lower_now)):
        return None

    direction_filter = BB_DIRECTION

    global _pending_break
    trend = None
    direction = None
    ref_price = None
    trigger_price = None
    band_ref = None
    break_ts = None

    # Si hay una rotura pendiente, esperar rebote (cierre del lado opuesto de la banda) en vela posterior
    if _pending_break:
        pend_dir = _pending_break.get("direction")
        break_ts = _pending_break.get("break_ts")
        if pend_dir == "long" and direction_filter != -1:
            if break_ts is not None and last_idx > break_ts and close_now > lower_now:
                trend = "alcista"
                direction = "long"
                # Entrada a precio de cierre de la vela de rebote; banda se mantiene como referencia
                ref_price = close_now
                trigger_price = close_now
                band_ref = lower_now
                print(
                    f"[ALERT][PENDING] Consumida rotura pendiente LONG (rebote) band={lower_now:.2f} close={close_now:.2f} ts={last_idx}"
                )
                _pending_break = None
        elif pend_dir == "short" and direction_filter != 1:
            if break_ts is not None and last_idx > break_ts and close_now < upper_now:
                trend = "bajista"
                direction = "short"
                ref_price = close_now
                trigger_price = close_now
                band_ref = upper_now
                print(
                    f"[ALERT][PENDING] Consumida rotura pendiente SHORT (rebote) band={upper_now:.2f} close={close_now:.2f} ts={last_idx}"
                )
                _pending_break = None
        # Si no se cumplió el rebote, seguimos esperando (no devolvemos alerta aún)

    # Si no hay alerta confirmada, registrar nuevas roturas
    if trend is None:
        # Rotura long: cierre por debajo de la banda inferior
        if close_now < lower_now and direction_filter != -1:
            if _pending_break and _pending_break.get("direction") != "long":
                print(
                    f"[ALERT][PENDING] Reset por cambio de tendencia (prev={_pending_break}); nueva LONG ts={last_idx} band={lower_now:.2f} close={close_now:.2f}"
                )
            print(
                f"[ALERT][PENDING] Set LONG ts={last_idx} band={lower_now:.2f} close={close_now:.2f} upper={upper_now:.2f} lower={lower_now:.2f}"
            )
            _pending_break = {"direction": "long", "band": lower_now, "break_ts": last_idx}
        # Rotura short: cierre por encima de la banda superior
        elif close_now > upper_now and direction_filter != 1:
            if _pending_break and _pending_break.get("direction") != "short":
                print(
                    f"[ALERT][PENDING] Reset por cambio de tendencia (prev={_pending_break}); nueva SHORT ts={last_idx} band={upper_now:.2f} close={close_now:.2f}"
                )
            print(
                f"[ALERT][PENDING] Set SHORT ts={last_idx} band={upper_now:.2f} close={close_now:.2f} upper={upper_now:.2f} lower={lower_now:.2f}"
            )
            _pending_break = {"direction": "short", "band": upper_now, "break_ts": last_idx}
        return None

    # Si llegamos aquí es porque se confirmó un rebote y se va a emitir señal

    stop_loss = None
    take_profit = None
    if ref_price and ref_price > 0:
        if STOP_LOSS_PCT > 0:
            stop_loss = ref_price * (1 - STOP_LOSS_PCT) if direction == "long" else ref_price * (1 + STOP_LOSS_PCT)
        if TAKE_PROFIT_PCT > 0:
            take_profit = ref_price * (1 + TAKE_PROFIT_PCT) if direction == "long" else ref_price * (1 - TAKE_PROFIT_PCT)

    last_bar = ohlc_stream.iloc[-1]
    bar_close_ts = last_bar.get("BarCloseTime", last_idx)
    volume = float(last_bar.get("Volume", np.nan))
    basis_now = float(basis.iloc[-1]) if basis is not None else np.nan

    timestamp = bar_close_ts if isinstance(bar_close_ts, pd.Timestamp) else pd.Timestamp(bar_close_ts)
    try:
        interval_delta = pd.to_timedelta(STREAM_INTERVAL)
    except (ValueError, TypeError):
        interval_delta = None

    return {
        "type": "bollinger_signal",
        "timestamp": (timestamp - interval_delta) if interval_delta is not None else timestamp,
        "message": (
            f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: Señal Bollinger {trend} en {trigger_price:.2f} "
            f"(banda de ruptura {band_ref:.2f} ref)"
        ),
        # Entrada al cierre de la vela de rebote (market); se deja banda como referencia
        "price": trigger_price,
        "entry_price": ref_price,
        "close_price": close_now,
        "direction": direction,
        "basis": basis_now,
        "reference_band": band_ref,
        "volume": volume,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        # Alias para TP/SL consumidos por watcher_alertas
        "sl": stop_loss,
        "tp": take_profit,
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

    base = f"{ts_str}\n{alert.get('message', '')}"

    sl = alert.get("stop_loss")
    tp = alert.get("take_profit")
    parts = []
    try:
        if tp is not None:
            parts.append(f"TP: {float(tp):.2f}")
        if sl is not None:
            parts.append(f"SL: {float(sl):.2f}")
    except Exception:
        pass

    if parts:
        base = f"{base}\n" + " | ".join(parts)

    return base


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
