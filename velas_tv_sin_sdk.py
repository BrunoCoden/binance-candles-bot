# velas_tv_sin_sdk.py
# ---------------------------------------------------------
# Velas Binance (USDⓈ-M)
# - Solo precios y velas (sin indicadores ni alertas)
# - Usa tu zona horaria local (UTC-3, Buenos Aires)
# ---------------------------------------------------------
import os, time
import numpy as np
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from binance.um_futures import UMFutures

# ================== Config ==================
load_dotenv()

SYMBOL_DISPLAY = os.getenv("SYMBOL", "ETHUSDT.P")
API_SYMBOL     = SYMBOL_DISPLAY.replace(".P", "")
INTERVAL       = os.getenv("INTERVAL", "30m")
LIMITE         = int(os.getenv("LIMITE", "1500"))
BASE_URL       = os.getenv("BASE_URL", "https://fapi.binance.com")  # mainnet
PRICE_SOURCE   = os.getenv("PRICE_SOURCE", "LAST").upper()          # LAST | MARK | INDEX

RUN_LOOP       = os.getenv("RUN_LOOP", "true").lower() in ("1","true","yes","y","on")
REFRESH_JITTER = int(os.getenv("REFRESH_JITTER_SEC", "2"))

DESPIKE_WITH_MARK = os.getenv("DESPIKE_WITH_MARK", "true").lower() in ("1","true","yes","y","on")
BAND_PCT          = float(os.getenv("BAND_PCT", "1.0"))

# ================== Zona horaria local ==================
LOCAL_TZ = ZoneInfo("America/Argentina/Buenos_Aires")

# ================== Cliente ==================
client = UMFutures(base_url=BASE_URL)

# ================== Utilidades ==================
def _interval_to_timedelta(interval: str) -> timedelta:
    s = interval.strip().lower()
    if s.endswith("m"): return timedelta(minutes=int(s[:-1]))
    if s.endswith("h"): return timedelta(hours=int(s[:-1]))
    if s.endswith("d"): return timedelta(days=int(s[:-1]))
    raise ValueError(f"Intervalo no soportado: {interval}")

def _next_close_after(ts: datetime, interval: str) -> datetime:
    dt = _interval_to_timedelta(interval)
    epoch = datetime(1970,1,1, tzinfo=LOCAL_TZ)
    secs  = int((ts - epoch).total_seconds())
    step  = int(dt.total_seconds())
    next_sec = ((secs // step) + 1) * step
    return epoch + timedelta(seconds=next_sec)

def _klines_to_df(raw):
    rows = []
    for k in raw:
        rows.append({
            "Datetime": datetime.fromtimestamp(k[0]/1000.0, LOCAL_TZ),
            "Open":  float(k[1]),
            "High":  float(k[2]),
            "Low":   float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]) if len(k) > 5 else 0.0,
        })
    df = pd.DataFrame(rows)
    return df.set_index("Datetime") if not df.empty else df

# ================== Fetch OHLC ==================
def get_prices_df(symbol=API_SYMBOL, interval=INTERVAL, limit=LIMITE) -> pd.DataFrame:
    if PRICE_SOURCE == "MARK":
        return _klines_to_df(client.mark_price_klines(symbol=symbol, interval=interval, limit=limit))
    if PRICE_SOURCE == "INDEX":
        return _klines_to_df(client.index_price_klines(symbol=symbol, interval=interval, limit=limit))

    last_df = _klines_to_df(client.klines(symbol=symbol, interval=interval, limit=limit))
    if last_df.empty or not DESPIKE_WITH_MARK:
        return last_df

    mark_df = _klines_to_df(client.mark_price_klines(symbol=symbol, interval=interval, limit=limit))
    if mark_df.empty:
        return last_df

    df = last_df.join(mark_df, lsuffix="_L", rsuffix="_M", how="inner")
    if df.empty:
        return last_df

    up_band = 1.0 + BAND_PCT/100.0
    dn_band = 1.0 - BAND_PCT/100.0

    adj_high = np.minimum(df["High_L"].values, df["High_M"].values * up_band)
    adj_low  = np.maximum(df["Low_L"].values,  df["Low_M"].values  * dn_band)

    o = df["Open_L"].values; c = df["Close_L"].values
    lo_floor   = np.minimum(o, c)
    hi_ceiling = np.maximum(o, c)
    adj_low    = np.minimum(adj_low,  hi_ceiling)
    adj_high   = np.maximum(adj_high, lo_floor)

    out = pd.DataFrame({
        "Open":   o,
        "High":   adj_high,
        "Low":    adj_low,
        "Close":  c,
        "Volume": df["Volume_L"].values,
    }, index=df.index)
    return out

# ================== Plot ==================
def plot_once():
    src = PRICE_SOURCE
    if PRICE_SOURCE == "LAST" and DESPIKE_WITH_MARK:
        src = f"LAST + DESPIKE(MARK ±{BAND_PCT}%)"

    print(f"Cargando {LIMITE} velas {INTERVAL} de {SYMBOL_DISPLAY} [{src}] ...")
    df = get_prices_df()
    if df.empty:
        print("Sin datos.")
        return None

    mpf.plot(
        df[["Open","High","Low","Close"]],
        type="candle",
        style="yahoo",
        tight_layout=True,
        warn_too_much_data=len(df)+1,
        title=f"{SYMBOL_DISPLAY} ({INTERVAL}) • {src} • Horario {LOCAL_TZ}"
    )
    return df

def run_loop():
    last_df = plot_once()
    last_close = last_df.index[-1] if last_df is not None and not last_df.empty else None

    try:
        while True:
            now = datetime.now(LOCAL_TZ)
            target = _next_close_after(now, INTERVAL)
            sleep_for = max(0.0, (target - now).total_seconds() + REFRESH_JITTER)
            time.sleep(sleep_for)

            df = get_prices_df()
            if df.empty:
                continue
            new_close = df.index[-1]
            if last_close is not None and new_close <= last_close:
                time.sleep(1.0)
                continue

            print(f"[Refresco] Nueva vela cerrada: {new_close}")
            mpf.plot(
                df[["Open","High","Low","Close"]],
                type="candle",
                style="yahoo",
                tight_layout=True,
                warn_too_much_data=len(df)+1,
                title=f"{SYMBOL_DISPLAY} ({INTERVAL}) • {PRICE_SOURCE} • Horario {LOCAL_TZ}"
            )
            last_close = new_close

    except KeyboardInterrupt:
        print("\nDetenido por el usuario. ¡Listo!")

# ================== Main ==================
def main():
    if RUN_LOOP:
        run_loop()
    else:
        plot_once()

if __name__ == "__main__":
    main()
