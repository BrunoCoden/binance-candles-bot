# soloAlertas.py
# ---------------------------------------------------------
# Velas + Canales Range Breakout (solo canales, sin buy/sell)
# - Bandas: SMA(ATR(200),100) * RB_MULTI (réplica Pine)
# - RB_INIT_BAR para alinear con TradingView (default 301)
# - Mid-lines grises (UpperMid/LowerMid) + Q-lines amarillas punteadas (UpperQ/LowerQ)
# - Señal de toque a Q-lines con punto ROJO y flags en CSV
# ---------------------------------------------------------

import os
import time
import numpy as np
import pandas as pd
import mplfinance as mpf
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

CSV_PATH       = os.getenv("CSV_PATH", "stream_canales.csv").strip()
SAVEFIG_PATH   = os.getenv("SAVEFIG", "").strip()
SLEEP_FALLBACK = int(os.getenv("SLEEP_FALLBACK", "10"))
WARN_TOO_MUCH  = 5000

BINANCE_INTERVAL_SECONDS = {
    "1m":60, "3m":180, "5m":300, "15m":900, "30m":1800, "1h":3600, "2h":7200,
    "4h":14400, "6h":21600, "8h":28800, "12h":43200, "1d":86400, "3d":259200,
    "1w":604800, "1M":2592000
}
def interval_seconds(s: str) -> int:
    return BINANCE_INTERVAL_SECONDS.get(s, SLEEP_FALLBACK)

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
            "CloseTime": int(k[6]),
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

    upper_q = (umid + vup) / 2.0   # Value + 3/4*width
    lower_q = (lmid + vlo) / 2.0   # Value - 3/4*width

    return pd.DataFrame({
        'Value': value,
        'ValueUpper': vup,
        'ValueLower': vlo,
        'UpperMid': umid,
        'LowerMid': lmid,
        'UpperQ': upper_q,
        'LowerQ': lower_q,
    }, index=df.index)

# ================== Helpers Plot ==================
def _linebreak_like(series: pd.Series) -> pd.Series:
    s = series.copy()
    prev = s.shift(1)
    s[(~prev.isna()) & (s != prev)] = np.nan
    return s

def _build_overlays(df: pd.DataFrame, chans: pd.DataFrame):
    # Toques a Q-lines: Low <= Q <= High
    touch_uq = (df['Low'] <= chans['UpperQ']) & (df['High'] >= chans['UpperQ'])
    touch_lq = (df['Low'] <= chans['LowerQ']) & (df['High'] >= chans['LowerQ'])
    suq = pd.Series(np.nan, index=df.index, dtype=float)
    slq = pd.Series(np.nan, index=df.index, dtype=float)
    suq.loc[touch_uq] = chans['UpperQ'].loc[touch_uq]
    slq.loc[touch_lq] = chans['LowerQ'].loc[touch_lq]

    return [
        # Canal principal
        mpf.make_addplot(_linebreak_like(chans['ValueUpper']), color='#1dac70', width=1),
        mpf.make_addplot(_linebreak_like(chans['Value']),      color='gray',    width=1),
        mpf.make_addplot(_linebreak_like(chans['ValueLower']), color='#df3a79', width=1),
        # Mid-lines en gris (se mantienen)
        mpf.make_addplot(_linebreak_like(chans['UpperMid']),   color='gray',    width=1, alpha=0.5),
        mpf.make_addplot(_linebreak_like(chans['LowerMid']),   color='gray',    width=1, alpha=0.5),
        # Q-lines: amarillas punteadas
        mpf.make_addplot(_linebreak_like(chans['UpperQ']),     color='yellow',  width=1, linestyle=':'),
        mpf.make_addplot(_linebreak_like(chans['LowerQ']),     color='yellow',  width=1, linestyle=':'),
        # Puntos ROJOS en los toques a Q-lines
        mpf.make_addplot(suq, type='scatter', marker='o', markersize=40, color='red'),
        mpf.make_addplot(slq, type='scatter', marker='o', markersize=40, color='red'),
    ]

def plot_with_overlays(df: pd.DataFrame, chans: pd.DataFrame, title="Canales RB"):
    ap = _build_overlays(df, chans)
    fig, _ = mpf.plot(
        df[["Open","High","Low","Close","Volume"]],
        type='candle',
        style=mpf.make_mpf_style(),
        addplot=ap, returnfig=True, figsize=(12,6),
        datetime_format='%Y-%m-%d %H:%M', title=title,
        warn_too_much_data=WARN_TOO_MUCH
    )
    if SAVEFIG_PATH:
        try:
            fig.savefig(SAVEFIG_PATH, dpi=130)
            print(f"[OK] Gráfico guardado en: {SAVEFIG_PATH}")
        except Exception as e:
            print(f"[WARN] No pude guardar figura: {e}")

# ================== CSV ==================
CSV_COLUMNS = [
    "CloseTimeMs","Date","Open","High","Low","Close","Volume",
    "Value","ValueUpper","ValueLower","UpperMid","LowerMid","UpperQ","LowerQ",
    "TouchUpperQ","TouchLowerQ"
]

def ensure_csv_header(path: str):
    if not os.path.exists(path):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(path, index=False, encoding="utf-8")

def append_row_to_csv(path: str, row: dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False, encoding="utf-8")

# ================== plot_from_csv ==================
def plot_from_csv():
    if not os.path.exists(CSV_PATH):
        print(f"[WARN] No existe CSV '{CSV_PATH}' todavía.")
        return
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    if df.empty:
        print("[WARN] CSV vacío.")
        return
    df = df.sort_values("CloseTimeMs").drop_duplicates(subset=["CloseTimeMs"], keep="last")
    df = df.set_index("Date")
    df = df.rename(columns={c:c.capitalize() for c in df.columns})

    for c in ["Open","High","Low","Close","Volume","Value","Valueupper","Valuelower",
              "Uppermid","Lowermid","Upperq","Lowerq","Touchupperq","Touchlowerq"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    ohlc = df[["Open","High","Low","Close","Volume"]]
    chans = df[["Value","Valueupper","Valuelower","Uppermid","Lowermid","Upperq","Lowerq"]].copy()
    chans.columns = ["Value","ValueUpper","ValueLower","UpperMid","LowerMid","UpperQ","LowerQ"]
    plot_with_overlays(ohlc, chans, title=f"{SYMBOL_DISPLAY} {INTERVAL} (desde CSV)")

# ================== Loop ==================
def run_loop():
    print(f"[INIT] {SYMBOL_DISPLAY} {INTERVAL} | TZ={TZ_NAME}")
    ensure_csv_header(CSV_PATH)

    last_logged_ms = None
    if os.path.exists(CSV_PATH):
        try:
            tail = pd.read_csv(CSV_PATH, usecols=["CloseTimeMs"]).tail(1)
            if not tail.empty:
                last_logged_ms = int(tail["CloseTimeMs"].iloc[0])
        except Exception:
            pass

    while True:
        try:
            df = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
            now_utc_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            df_closed = df[df["CloseTime"] <= now_utc_ms]
            if df_closed.empty:
                time.sleep(SLEEP_FALLBACK)
                continue

            ohlc  = df_closed[["Open","High","Low","Close","Volume"]]
            chans = compute_channels(ohlc, multi=RB_MULTI, init_bar=RB_INIT_BAR)

            last_row = df_closed.iloc[-1]
            last_idx = df_closed.index[-1]
            last_ms  = int(last_row["CloseTime"])

            if (last_logged_ms is None) or (last_ms > last_logged_ms):
                c = chans.loc[last_idx]

                touch_uq = int((last_row["Low"] <= c["UpperQ"] <= last_row["High"]) if pd.notna(c["UpperQ"]) else 0)
                touch_lq = int((last_row["Low"] <= c["LowerQ"] <= last_row["High"]) if pd.notna(c["LowerQ"]) else 0)

                if touch_uq:
                    print(f"[TOUCH UpperQ] {fmt_ts(last_idx)} {SYMBOL_DISPLAY} {INTERVAL} | {round(float(c['UpperQ']), 6)}")
                if touch_lq:
                    print(f"[TOUCH LowerQ] {fmt_ts(last_idx)} {SYMBOL_DISPLAY} {INTERVAL} | {round(float(c['LowerQ']), 6)}")

                row = {
                    "CloseTimeMs": last_ms,
                    "Date":   fmt_ts(last_idx),
                    "Open":   round(float(last_row["Open"]),  6),
                    "High":   round(float(last_row["High"]),  6),
                    "Low":    round(float(last_row["Low"]),   6),
                    "Close":  round(float(last_row["Close"]), 6),
                    "Volume": round(float(last_row["Volume"]),6),
                    "Value":      float(c["Value"])      if pd.notna(c["Value"]) else np.nan,
                    "ValueUpper": float(c["ValueUpper"]) if pd.notna(c["ValueUpper"]) else np.nan,
                    "ValueLower": float(c["ValueLower"]) if pd.notna(c["ValueLower"]) else np.nan,
                    "UpperMid":   float(c["UpperMid"])   if pd.notna(c["UpperMid"]) else np.nan,
                    "LowerMid":   float(c["LowerMid"])   if pd.notna(c["LowerMid"]) else np.nan,
                    "UpperQ":     float(c["UpperQ"])     if pd.notna(c["UpperQ"]) else np.nan,
                    "LowerQ":     float(c["LowerQ"])     if pd.notna(c["LowerQ"]) else np.nan,
                    "TouchUpperQ": touch_uq,
                    "TouchLowerQ": touch_lq,
                }
                append_row_to_csv(CSV_PATH, row)

                print(f"[{row['Date']}] O:{row['Open']:>10} H:{row['High']:>10} L:{row['Low']:>10} C:{row['Close']:>10} "
                      f"| Up:{row['ValueUpper']:>10}  UMid:{row['UpperMid']:>10}  UQ:{row['UpperQ']:>10}  "
                      f"Lo:{row['ValueLower']:>10}  LMid:{row['LowerMid']:>10}  LQ:{row['LowerQ']:>10}")

                last_logged_ms = last_ms

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
    print(f"[INFO] Loop de velas (canales + Q-lines amarillas) → CSV='{CSV_PATH}'")
    run_loop()

if __name__ == "__main__":
    main()
