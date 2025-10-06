# soloAlertas.py
# ---------------------------------------------------------
# Velas + Canales Range Breakout (solo canales, sin buy/sell)
# Salida ÚNICA: tabla.csv (una fila por vela cerrada)
# Columns: CloseTimeMs, Date, Open, High, Low, Close, Volume, TouchUpperQ, TouchLowerQ
# ---------------------------------------------------------

import os
import time
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
from dotenv import load_dotenv

# ================== Config/Entorno ==================
load_dotenv()

SYMBOL_DISPLAY = os.getenv("SYMBOL", "ETHUSDT.P")
API_SYMBOL     = SYMBOL_DISPLAY.replace(".P", "")
INTERVAL       = os.getenv("INTERVAL", "30m")
LIMIT          = int(os.getenv("LIMIT", "800"))
TZ_NAME        = os.getenv("TZ", "America/Argentina/Buenos_Aires")

RB_MULTI       = float(os.getenv("RB_MULTI", "4.0"))
RB_INIT_BAR    = int(os.getenv("RB_INIT_BAR", "301"))

# ÚNICA salida CSV
TABLE_CSV_PATH = os.getenv("TABLE_CSV_PATH", "tabla.csv").strip()
TABLE_COLUMNS  = [
    "CloseTimeMs","Date","Open","High","Low","Close","Volume",
    "TouchUpperQ","TouchLowerQ"
]

SLEEP_FALLBACK = int(os.getenv("SLEEP_FALLBACK", "10"))

def fmt_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")

# ================== Binance client ==================
try:
    from binance.um_futures import UMFutures
except Exception:
    print("ERROR: Falta el conector de Futuros de Binance.")
    print("Instalá con:  pip install binance-futures-connector")
    raise

def get_binance_client():
    base_url = os.getenv("BINANCE_UM_BASE_URL", "https://fapi.binance.com")
    return UMFutures(base_url=base_url)

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    client = get_binance_client()
    data = client.klines(symbol=symbol, interval=interval, limit=limit)
    rows = []
    for k in data:
        rows.append({
            "OpenTime": int(k[0]),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
            "CloseTime": int(k[6]),  # fin de vela en ms (UTC)
        })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df["CloseTimeDT"] = pd.to_datetime(df["CloseTime"], unit="ms", utc=True)
    tz = ZoneInfo(TZ_NAME)
    df = df.set_index(df["Date"].dt.tz_convert(tz)).sort_index()
    df["CloseTimeDT"] = df["CloseTimeDT"].dt.tz_convert(tz)
    return df[["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"]]

# ================== Indicador: SOLO canales ==================
def _rma(x: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    return x.ewm(alpha=alpha, adjust=False).mean()

def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return _rma(tr, length)

def compute_channels(df: pd.DataFrame, multi: float = 4.0, init_bar: int = 301) -> pd.DataFrame:
    """
    Devuelve DataFrame con:
      Value, ValueUpper, ValueLower, UpperMid, LowerMid, UpperQ, LowerQ
    UpperMid = (Value + ValueUpper)/2
    LowerMid = (Value + ValueLower)/2
    UpperQ   = (UpperMid + ValueUpper)/2 = Value + 3/4*width
    LowerQ   = (LowerMid + ValueLower)/2 = Value - 3/4*width
    """
    df = df.copy()
    df['hl2'] = (df['High'] + df['Low']) / 2.0

    atr200 = _atr(df, 200)
    width  = atr200.rolling(100, min_periods=1).mean() * multi

    n = len(df)
    value = np.full(n, np.nan)
    vup   = np.full(n, np.nan)
    vlo   = np.full(n, np.nan)
    umid  = np.full(n, np.nan)
    lmid  = np.full(n, np.nan)

    highs = df['High'].values
    lows  = df['Low'].values
    hl2   = df['hl2'].values
    w     = width.values

    def crossed_up(prev_val, curr_val, prev_lvl, curr_lvl):
        return (prev_val <= prev_lvl) and (curr_val > curr_lvl)

    def crossed_dn(prev_val, curr_val, prev_lvl, curr_lvl):
        return (prev_val >= prev_lvl) and (curr_val < curr_lvl)

    count = 0

    for i in range(n):
        if i == init_bar:
            value[i] = hl2[i]
            vup[i]   = hl2[i] + w[i]
            vlo[i]   = hl2[i] - w[i]
            umid[i]  = (value[i] + vup[i]) / 2.0
            lmid[i]  = (value[i] + vlo[i]) / 2.0
        else:
            if i > 0:
                value[i] = value[i-1]
                vup[i]   = vup[i-1]
                vlo[i]   = vlo[i-1]
                umid[i]  = umid[i-1]
                lmid[i]  = lmid[i-1]

        if i < max(init_bar, 1):
            continue

        cross_up   = crossed_up(lows[i-1],  lows[i],  vup[i-1], vup[i])
        cross_down = crossed_dn(highs[i-1], highs[i], vlo[i-1], vlo[i])

        if not np.isnan(vup[i]) and not np.isnan(vlo[i]):
            if (lows[i] > vup[i]) or (highs[i] < vlo[i]):
                count += 1

        if cross_up or cross_down or (count == 100):
            count   = 0
            value[i] = hl2[i]
            vup[i]   = hl2[i] + w[i]
            vlo[i]   = hl2[i] - w[i]
            umid[i]  = (value[i] + vup[i]) / 2.0
            lmid[i]  = (value[i] + vlo[i]) / 2.0

    upper_q = (umid + vup) / 2.0
    lower_q = (lmid + vlo) / 2.0

    return pd.DataFrame({
        'Value': value,
        'ValueUpper': vup,
        'ValueLower': vlo,
        'UpperMid': umid,
        'LowerMid': lmid,
        'UpperQ': upper_q,
        'LowerQ': lower_q,
    }, index=df.index)

# ================== CSV: única salida tabla.csv ==================
def ensure_table_csv_header(path: str):
    if not os.path.exists(path):
        pd.DataFrame(columns=TABLE_COLUMNS).to_csv(path, index=False, encoding="utf-8")

def append_row_to_table(path: str, row: dict):
    out = {k: row.get(k, np.nan) for k in TABLE_COLUMNS}
    pd.DataFrame([out]).to_csv(path, mode="a", header=False, index=False, encoding="utf-8")

# ================== Loop ==================
def run_loop():
    print(f"[INIT] {SYMBOL_DISPLAY} {INTERVAL} | TZ={TZ_NAME}")
    ensure_table_csv_header(TABLE_CSV_PATH)

    last_logged_ms = None
    if os.path.exists(TABLE_CSV_PATH):
        try:
            tail = pd.read_csv(TABLE_CSV_PATH, usecols=["CloseTimeMs"]).tail(1)
            if not tail.empty:
                last_logged_ms = int(tail["CloseTimeMs"].iloc[0])
        except Exception:
            pass

    while True:
        try:
            df = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
            now_utc_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            # Solo velas CERRADAS
            df_closed = df[df["CloseTime"] <= now_utc_ms]
            if df_closed.empty:
                time.sleep(SLEEP_FALLBACK)
                continue

            ohlc  = df_closed[["Open","High","Low","Close","Volume"]]
            chans = compute_channels(ohlc, multi=RB_MULTI, init_bar=RB_INIT_BAR)

            last_row = df_closed.iloc[-1]
            last_idx = df_closed.index[-1]        # tz-aware
            last_ms  = int(last_row["CloseTime"]) # clave numérica

            if (last_logged_ms is None) or (last_ms > last_logged_ms):
                c = chans.loc[last_idx]

                # Señales: toque a Q-lines (cuartiles) en la vela cerrada
                touch_uq = int((last_row["Low"] <= c["UpperQ"] <= last_row["High"]) if pd.notna(c["UpperQ"]) else 0)
                touch_lq = int((last_row["Low"] <= c["LowerQ"] <= last_row["High"]) if pd.notna(c["LowerQ"]) else 0)

                trow = {
                    "CloseTimeMs": last_ms,
                    "Date":   fmt_ts(last_idx),
                    "Open":   round(float(last_row["Open"]),  6),
                    "High":   round(float(last_row["High"]),  6),
                    "Low":    round(float(last_row["Low"]),   6),
                    "Close":  round(float(last_row["Close"]), 6),
                    "Volume": round(float(last_row["Volume"]),6),
                    "TouchUpperQ": touch_uq,
                    "TouchLowerQ": touch_lq,
                }
                append_row_to_table(TABLE_CSV_PATH, trow)

                # Log humano, porque nos gusta sufrir viendo números
                print(f"[{trow['Date']}] O:{trow['Open']:>10} H:{trow['High']:>10} L:{trow['Low']:>10} C:{trow['Close']:>10} "
                      f"| UpperQ_touch:{trow['TouchUpperQ']}  LowerQ_touch:{trow['TouchLowerQ']}")

                last_logged_ms = last_ms

            # Dormir hasta el próximo cierre de vela estimado por Binance
            next_close_ms = int(df.iloc[-1]["CloseTime"])
            now_utc = datetime.now(timezone.utc).timestamp()
            eta = max(2, int((next_close_ms/1000) - now_utc) + 1)
            time.sleep(eta)

        except KeyboardInterrupt:
            print("\n[EXIT] Cortado por usuario.")
            break
        except Exception as e:
            print(f"\n[WARN] {type(e).__name__}: {e}")
            time.sleep(SLEEP_FALLBACK)

# ================== Main ==================
def main():
    print(f"[INFO] Loop de velas → CSV único '{TABLE_CSV_PATH}'")
    run_loop()

if __name__ == "__main__":
    main()
