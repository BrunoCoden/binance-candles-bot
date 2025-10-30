import os

import numpy as np
import pandas as pd
import mplfinance as mpf

from paginado_binance import fetch_klines_paginado
from velas import (
    SYMBOL_DISPLAY,
    API_SYMBOL,
    STREAM_INTERVAL,
    BB_LENGTH,
    BB_MULT,
    compute_bollinger_bands,
)


PLOT_STREAM_BARS = int(os.getenv("PLOT_STREAM_BARS", "5000"))
WARN_TOO_MUCH = int(os.getenv("WARN_TOO_MUCH", "5000"))
BB_LINE_WIDTH = float(os.getenv("BB_LINE_WIDTH", "2.0"))
BB_BASIS_COLOR = os.getenv("BB_BASIS_COLOR", "#facc15")
BB_UPPER_COLOR = os.getenv("BB_UPPER_COLOR", "#1dac70")
BB_LOWER_COLOR = os.getenv("BB_LOWER_COLOR", "#dc2626")
BB_FILL_ALPHA = float(os.getenv("BB_FILL_ALPHA", "0.12"))


def _style_tv_dark():
    mc = mpf.make_marketcolors(
        up="lime",
        down="red",
        edge="inherit",
        wick="white",
        volume="in",
    )
    return mpf.make_mpf_style(
        marketcolors=mc,
        base_mpf_style="nightclouds",
        facecolor="black",
        edgecolor="black",
        gridcolor="#333333",
        gridstyle="--",
        rc={"axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"},
        y_on_right=False,
    )


def _has_data(s: pd.Series | None) -> bool:
    if s is None or len(s) == 0:
        return False
    try:
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        if arr.size == 0:
            return False
        return np.isfinite(arr).any()
    except Exception:
        return False


def main():
    df_stream = fetch_klines_paginado(API_SYMBOL, STREAM_INTERVAL, PLOT_STREAM_BARS)
    if df_stream.empty:
        raise SystemExit("[ERROR] No se pudieron obtener velas del stream")

    ohlc_stream = df_stream[["Open", "High", "Low", "Close", "Volume"]]
    bb = compute_bollinger_bands(ohlc_stream, BB_LENGTH, BB_MULT).reindex(ohlc_stream.index).ffill()

    basis = bb.get("basis")
    upper = bb.get("upper")
    lower = bb.get("lower")

    addplots = []

    if _has_data(basis):
        addplots.append(mpf.make_addplot(basis, color=BB_BASIS_COLOR, width=BB_LINE_WIDTH, ylabel="Bollinger"))
    if _has_data(upper):
        addplots.append(mpf.make_addplot(upper, color=BB_UPPER_COLOR, width=BB_LINE_WIDTH, linestyle="--"))
    if _has_data(lower):
        addplots.append(mpf.make_addplot(lower, color=BB_LOWER_COLOR, width=BB_LINE_WIDTH, linestyle="--"))

    if _has_data(upper) and _has_data(lower):
        upper_vals = pd.to_numeric(upper, errors="coerce").to_numpy(dtype="float64")
        lower_vals = pd.to_numeric(lower, errors="coerce").to_numpy(dtype="float64")
        mask = np.isfinite(upper_vals) & np.isfinite(lower_vals)
        if mask.any():
            addplots.append(
                mpf.make_addplot(
                    upper,
                    color=BB_UPPER_COLOR,
                    width=0,
                    fill_between=dict(
                        y1=upper_vals,
                        y2=lower_vals,
                        where=mask,
                        alpha=BB_FILL_ALPHA,
                        color=BB_UPPER_COLOR,
                    ),
                )
            )

    mpf.plot(
        ohlc_stream,
        type="candle",
        style=_style_tv_dark(),
        addplot=addplots,
        figsize=(12, 6),
        datetime_format="%Y-%m-%d %H:%M",
        title=f"{SYMBOL_DISPLAY} {STREAM_INTERVAL} — Bandas de Bollinger",
        warn_too_much_data=WARN_TOO_MUCH,
    )


if __name__ == "__main__":
    main()
