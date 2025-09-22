# -*- coding: utf-8 -*-
import argparse
import sys
from typing import List, Any

import requests
import pandas as pd
import mplfinance as mpf
import pytz  # para manejar zona horaria local

SUPPORTED: List[str] = [
    "1s", "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"
]

def fetch(mode: str, symbol: str, interval: str, limit: int, price_type: str, testnet: bool) -> Any:
    if interval not in SUPPORTED:
        raise ValueError(f"Intervalo no soportado: {interval}")
    symbol = symbol.upper()

    if mode == "spot":
        base = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
        url = f"{base}/api/v3/klines"
    else:
        base = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
        if price_type == "mark":
            url = f"{base}/fapi/v1/markPriceKlines"
        elif price_type == "index":
            url = f"{base}/fapi/v1/indexPriceKlines"
        else:  # last
            url = f"{base}/fapi/v1/klines"

    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and "code" in data:
        raise RuntimeError(f"Binance error {data.get('code')}: {data.get('msg')} ({url})")

    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError("La respuesta no contiene velas (lista vacía).")

    return data

def to_df(kl):
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"
    ]
    df = pd.DataFrame(kl, columns=cols)

    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", errors="coerce", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", errors="coerce", utc=True)
    df = df.dropna(subset=["open_time"])

    # aplicar zona horaria local (Corrientes: America/Argentina/Buenos_Aires)
    local_tz = "America/Argentina/Buenos_Aires"
    df["open_time"]  = df["open_time"].dt.tz_convert(local_tz)
    df["close_time"] = df["close_time"].dt.tz_convert(local_tz)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.set_index("open_time")
    df.index.name = "Date"
    df = df[["open", "high", "low", "close"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if df.empty:
        raise RuntimeError("DataFrame resultante vacío después de limpiar. No hay velas válidas.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("El índice no es DatetimeIndex.")
    return df

def style():
    return mpf.make_mpf_style(
        base_mpl_style="classic",
        marketcolors=mpf.make_marketcolors(up="green", down="red", edge="inherit", wick="inherit")
    )

def main():
    ap = argparse.ArgumentParser("Velas Binance estilo TradingView (REST robusto)")
    ap.add_argument("--mode", choices=["spot", "futures"], default="futures")
    ap.add_argument("--testnet", action="store_true")
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--interval", default="30m")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--price-type", choices=["last","mark","index"], default="mark",
                    help="En Futuros: tipo de precio (last, mark, index). Por defecto: mark.")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()

    try:
        raw = fetch(a.mode, a.symbol, a.interval, a.limit, a.price_type, a.testnet)
        df = to_df(raw)

        ref = a.price_type.upper() if a.mode == "futures" else "LAST"
        env = "TESTNET" if a.testnet else "PROD"
        ttl = f"{a.symbol.upper()} • {a.interval} • {ref} • {env}"

        if a.png:
            mpf.plot(
                df, type="candle", style=style(),
                title=ttl, tight_layout=True,
                savefig=dict(fname=a.png, dpi=150, bbox_inches="tight")
            )
            print(f"PNG guardado en: {a.png}")
        else:
            mpf.plot(
                df, type="candle", style=style(),
                title=ttl, tight_layout=True
            )

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
