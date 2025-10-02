# grafico_live.py
# ---------------------------------------------------------
# Dibuja velas + Range Breakout en vivo (sin tocar el loop).
# - Baja datos de Binance
# - Reusa funciones de velas_tv_sin_sdk.py
# ---------------------------------------------------------

import os
from dotenv import load_dotenv
import mplfinance as mpf
from velas_tv_sin_sdk import get_binance_client, fetch_klines, compute_range_breakout, _build_overlays

# Cargar .env
load_dotenv()

# Configuración (se toma igual que en velas_tv_sin_sdk.py)
SYMBOL_DISPLAY   = os.getenv("SYMBOL", "ETHUSDT.P")
API_SYMBOL       = SYMBOL_DISPLAY.replace(".P", "")
INTERVAL         = os.getenv("INTERVAL", "30m")   # ← corregido
LIMIT            = int(os.getenv("LIMIT", "500"))

RB_MULTI         = float(os.getenv("RB_MULTI", "4.0"))
RB_LB            = int(os.getenv("RB_LB", "10"))
RB_FILTER_TREND  = os.getenv("RB_FILTER_TREND", "false").lower() == "true"

def plot_live():
    """Descarga velas desde Binance y grafica el Range Breakout con señales."""
    df = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
    indi = compute_range_breakout(df[["Open","High","Low","Close","Volume"]],
                                  multi=RB_MULTI, lb=RB_LB, filter_trend=RB_FILTER_TREND)

    ap = _build_overlays(df, indi)
    mpf.plot(df[["Open","High","Low","Close","Volume"]],
             type="candle", style=mpf.make_mpf_style(),
             addplot=ap, figsize=(12,6),
             datetime_format="%Y-%m-%d %H:%M",
             title=f"{SYMBOL_DISPLAY} {INTERVAL} (live)")

if __name__ == "__main__":
    plot_live()
