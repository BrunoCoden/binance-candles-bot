# velas_TV_sin_sdk.py
# ---------------------------------------------------------
# Velas + Range Breakout (canal + flechas, sin círculos)
# - Lee variables desde .env (load_dotenv)
# - Loop infinito: detecta vela cerrada y la guarda en CSV
# - Salida por consola formateada
# - Gráfico opcional una vez al inicio
# ---------------------------------------------------------

import os
import time
import math
import numpy as np
import pandas as pd
import mplfinance as mpf
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
from dotenv import load_dotenv  # <-- .env

# Cargar variables desde .env (si existe)
load_dotenv()

try:
    from binance.um_futures import UMFutures
except Exception:
    print("ERROR: Falta el conector de Futuros de Binance.")
    print("Instalá con:  pip install binance-futures-connector")
    raise

# ================== Config (desde .env con defaults) ==================
SYMBOL_DISPLAY   = os.getenv("SYMBOL", "ETHUSDT.P")
API_SYMBOL       = SYMBOL_DISPLAY.replace(".P", "")
INTERVAL         = os.getenv("INTERVAL", "30m")          # 1m 3m 5m 15m 30m 1h ...
LIMIT            = int(os.getenv("LIMIT", "800"))        # <= 1200 en Binance
TZ_NAME          = os.getenv("TZ", "America/Argentina/Buenos_Aires")

RB_MULTI         = float(os.getenv("RB_MULTI", "4.0"))
RB_LB            = int(os.getenv("RB_LB", "10"))
RB_FILTER_TREND  = os.getenv("RB_FILTER_TREND", "false").lower() == "true"

CSV_PATH         = os.getenv("CSV_PATH", "stream_table.csv").strip()
SHOW_ON_START    = os.getenv("SHOW_ON_START", "0") != "0"  # por defecto 0 (evita bloqueos)
SAVEFIG_PATH     = os.getenv("SAVEFIG", "").strip()
SLEEP_FALLBACK   = int(os.getenv("SLEEP_FALLBACK", "10"))

WARN_TOO_MUCH    = 5000

# Intervalos de Binance en segundos (para dormir hasta próximo cierre)
BINANCE_INTERVAL_SECONDS = {
    "1m":60, "3m":180, "5m":300, "15m":900, "30m":1800, "1h":3600, "2h":7200,
    "4h":14400, "6h":21600, "8h":28800, "12h":43200, "1d":86400, "3d":259200,
    "1w":604800, "1M":2592000
}
def interval_seconds(s: str) -> int:
    return BINANCE_INTERVAL_SECONDS.get(s, SLEEP_FALLBACK)

def fmt_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")

# ================== Datos ==================
def get_binance_client():
    base_url = os.getenv("BINANCE_UM_BASE_URL", "https://fapi.binance.com")
    return UMFutures(base_url=base_url)

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Obtiene klines y devuelve DataFrame con índices y timestamps en tu TZ + CloseTime/DT."""
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
            "CloseTime": int(k[6]),  # FIN DE VELA (ms, UTC)
        })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df["CloseTimeDT"] = pd.to_datetime(df["CloseTime"], unit="ms", utc=True)
    tz = ZoneInfo(TZ_NAME)
    df = df.set_index(df["Date"].dt.tz_convert(tz)).sort_index()
    df["CloseTimeDT"] = df["CloseTimeDT"].dt.tz_convert(tz)
    return df[["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"]]

# ================== Indicador ==================
def _rma(x: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / length
    return x.ewm(alpha=alpha, adjust=False).mean()

def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return _rma(tr, length)

def compute_range_breakout(df: pd.DataFrame,
                           multi: float = 4.0,
                           lb: int = 10,
                           filter_trend: bool = False):
    """Replica la lógica del Pine: canal + medias intermedias + flechas; sin círculos."""
    df = df.copy()
    df['hl2'] = (df['High'] + df['Low']) / 2.0
    atr200 = _atr(df, 200)
    atr_smooth = atr200.rolling(100, min_periods=1).mean()
    width = atr_smooth * multi

    n = len(df)
    value = np.full(n, np.nan)
    vup   = np.full(n, np.nan)
    vlo   = np.full(n, np.nan)
    umid  = np.full(n, np.nan)
    lmid  = np.full(n, np.nan)
    trend = np.full(n, False)
    plot_buy  = np.full(n, False)
    plot_sell = np.full(n, False)

    count = 0
    t = False
    arrows_buy, arrows_sell = [], []

    highs = df['High'].values
    lows  = df['Low'].values
    hl2   = df['hl2'].values
    w     = width.values
    idx   = df.index

    def crossover(low_now, level_now, low_prev):
        return (low_prev <= level_now) and (low_now > level_now)

    def crossunder(high_now, level_now, high_prev):
        return (high_prev >= level_now) and (high_now < level_now)

    for i in range(n):
        # Inicializa canal cuando hay historial suficiente (bar_index==301 en Pine)
        if i == 301:
            value[i] = hl2[i]
            vup[i]   = hl2[i] + w[i]
            vlo[i]   = hl2[i] - w[i]
            umid[i]  = (value[i] + vup[i]) / 2.0
            lmid[i]  = (value[i] + vlo[i]) / 2.0
        else:
            if i > 0:
                value[i] = value[i-1]; vup[i] = vup[i-1]; vlo[i] = vlo[i-1]
                umid[i] = umid[i-1]; lmid[i] = lmid[i-1]

        if i < 301:
            trend[i] = t
            continue

        low_prev, high_prev = lows[i-1], highs[i-1]
        cross_up   = crossover(lows[i], vup[i], low_prev)     # ruptura al techo → buy
        cross_down = crossunder(highs[i], vlo[i], high_prev)  # ruptura al piso  → sell

        # Conteo de barras fuera de canal
        if not np.isnan(vup[i]) and not np.isnan(vlo[i]):
            if lows[i] > vup[i] or highs[i] < vlo[i]:
                count += 1

        # Reset del canal
        channel_changed = False
        if (cross_up or cross_down or count == 100):
            count = 0
            value[i] = hl2[i]; vup[i] = hl2[i] + w[i]; vlo[i] = hl2[i] - w[i]
            umid[i] = (value[i] + vup[i]) / 2.0; lmid[i] = (value[i] + vlo[i]) / 2.0
            channel_changed = True

        # Actualiza tendencia
        if cross_up:
            t = True
        if cross_down:
            t = False

        trend[i] = t
        chage = not channel_changed  # true si el canal NO cambió en esta vela

        # Flechas por cruce de medias (en i-1, estilo Pine)
        if chage and not np.isnan(lmid[i]) and not np.isnan(umid[i]):
            buy_cross  = (lows[i-1]  <= lmid[i]) and (lows[i]  > lmid[i])
            sell_cross = (highs[i-1] >= umid[i]) and (highs[i] < umid[i])

            lb_ok_buy  = (lb == 0 or (i - lb >= 0 and lows[i - lb]  > lmid[i]))
            lb_ok_sell = (lb == 0 or (i - lb >= 0 and highs[i - lb] < umid[i]))

            if buy_cross and lb_ok_buy and (t if filter_trend else True):
                plot_buy[i] = True
                arrows_buy.append((idx[i-1], lows[i-1]))
            if sell_cross and lb_ok_sell and ((not t) if filter_trend else True):
                plot_sell[i] = True
                arrows_sell.append((idx[i-1], highs[i-1]))

        # Flechas por ruptura (en i)
        if cross_up and (t if filter_trend else True):
            plot_buy[i] = True
            arrows_buy.append((idx[i], lows[i]))
        if cross_down and ((not t) if filter_trend else True):
            plot_sell[i] = True
            arrows_sell.append((idx[i], highs[i]))

    buy_flag  = pd.Series(plot_buy,  index=df.index)
    sell_flag = pd.Series(plot_sell, index=df.index)

    return {
        'value': pd.Series(value, index=df.index),
        'value_upper': pd.Series(vup, index=df.index),
        'value_lower': pd.Series(vlo, index=df.index),
        'upper_mid': pd.Series(umid, index=df.index),
        'lower_mid': pd.Series(lmid, index=df.index),
        'buy_flag': buy_flag,
        'sell_flag': sell_flag,
        'arrows_buy': arrows_buy,
        'arrows_sell': arrows_sell,
    }

# ================== CSV ==================
CSV_COLUMNS = ["Date","Open","High","Low","Close","Volume","Buy","Sell"]

def ensure_csv_header(path: str):
    if not os.path.exists(path):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(path, index=False, encoding="utf-8")

def append_row_to_csv(path: str, row: dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False, encoding="utf-8")

# ================== Loop ==================
def run_loop():
    print(f"[INIT] {SYMBOL_DISPLAY} {INTERVAL} | TZ={TZ_NAME} | multi={RB_MULTI} LB={RB_LB} FILTER_TREND={int(RB_FILTER_TREND)}")
    ensure_csv_header(CSV_PATH)

    # Si ya existe CSV, recordamos la última vela registrada para no duplicar
    last_logged_ts = None
    if os.path.exists(CSV_PATH):
        try:
            tail = pd.read_csv(CSV_PATH).tail(1)
            if not tail.empty:
                last_logged_ts = pd.to_datetime(tail["Date"].iloc[0])
                if last_logged_ts.tzinfo is None:
                    last_logged_ts = last_logged_ts.replace(tzinfo=ZoneInfo(TZ_NAME))
        except Exception:
            pass

    sec = interval_seconds(INTERVAL)
    first_plot_done = False

    while True:
        try:
            # Traer klines
            df = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
            now_utc_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            # Sólo velas CERRADAS
            df_closed = df[df["CloseTime"] <= now_utc_ms]
            if df_closed.empty:
                time.sleep(SLEEP_FALLBACK)
                continue

            # Última vela cerrada
            last_row = df_closed.iloc[-1]
            last_ts  = df_closed.index[-1]  # en tu TZ

            # Loggear solo si es una vela nueva
            if (last_logged_ts is None) or (last_ts > last_logged_ts):
                indi = compute_range_breakout(df_closed[["Open","High","Low","Close","Volume"]],
                                              multi=RB_MULTI, lb=RB_LB, filter_trend=RB_FILTER_TREND)

                buy_flag  = bool(indi['buy_flag'].iloc[-1])
                sell_flag = bool(indi['sell_flag'].iloc[-1])

                row = {
                    "Date":   fmt_ts(last_ts),
                    "Open":   round(float(last_row["Open"]), 2),
                    "High":   round(float(last_row["High"]), 2),
                    "Low":    round(float(last_row["Low"]), 2),
                    "Close":  round(float(last_row["Close"]), 2),
                    "Volume": round(float(last_row["Volume"]), 2),
                    "Buy":    int(buy_flag),
                    "Sell":   int(sell_flag),
                }
                append_row_to_csv(CSV_PATH, row)

                sig = "▲ BUY" if buy_flag and not sell_flag else "▼ SELL" if sell_flag and not buy_flag else " "
                print(f"[{row['Date']}]  O:{row['Open']:>8}  H:{row['High']:>8}  L:{row['Low']:>8}  C:{row['Close']:>8}  Vol:{row['Volume']:>10}   Sig:{sig}")

                # Gráfico opcional 1 vez
                if SHOW_ON_START and not first_plot_done:
                    try:
                        fig, ax = mpf.plot(
                            df_closed[["Open","High","Low","Close","Volume"]],
                            type='candle', style=mpf.make_mpf_style(),
                            returnfig=True, figsize=(12,6),
                            datetime_format='%Y-%m-%d %H:%M'
                        )
                        if SAVEFIG_PATH:
                            fig.savefig(SAVEFIG_PATH, dpi=130)
                            print(f"[OK] Gráfico guardado en: {SAVEFIG_PATH}")
                        # Mostrar sin bloquear, si el backend lo permite
                        try:
                            mpf.show(block=False)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    first_plot_done = True

                last_logged_ts = last_ts

            # Dormir hasta el próximo cierre de vela
            next_close_ms = int(df.iloc[-1]["CloseTime"])  # de la vela en formación
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
    print(f"[INFO] Loop de velas → CSV='{CSV_PATH}'")
    run_loop()

if __name__ == "__main__":
    main()
