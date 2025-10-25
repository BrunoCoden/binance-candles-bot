# grafico_de_script_principal_solo_bandas.py
import os
import numpy as np
import pandas as pd
import mplfinance as mpf

from script_principal_de_velas_solo_bandas import (
    compute_channels,
    SYMBOL_DISPLAY, API_SYMBOL,
    CHANNEL_INTERVAL, STREAM_INTERVAL,
    RB_MULTI, RB_INIT_BAR
)

# NUEVO: paginador
from paginado_binance import fetch_klines_paginado

# Cuántas velas pedir (configurable por .env)
PLOT_STREAM_BARS  = int(os.getenv("PLOT_STREAM_BARS",  "5000"))   # 1m → ~3.5 días
PLOT_CHANNEL_BARS = int(os.getenv("PLOT_CHANNEL_BARS", "2000"))   # 30m → ~41 días
WARN_TOO_MUCH = int(os.getenv("WARN_TOO_MUCH", "5000"))
SUPER_ATR_PERIOD = int(os.getenv("SUPER_ATR_PERIOD", "10"))
SUPER_FACTOR     = float(os.getenv("SUPER_FACTOR", "3.0"))
SUPER_LINE_WIDTH = float(os.getenv("SUPER_LINE_WIDTH", "2.6"))
COLOR_BULL_LINE  = os.getenv("COLOR_BULL_LINE",  "#1dac70")
COLOR_BEAR_LINE  = os.getenv("COLOR_BEAR_LINE",  "#df3a79")
CHANNEL_FILL_ALPHA = float(os.getenv("CHANNEL_FILL_ALPHA", "0.18"))
SUPER_FILL_ALPHA   = float(os.getenv("SUPER_FILL_ALPHA", "0.12"))

def _linebreak_like(s: pd.Series) -> pd.Series:
    s = s.copy()
    prev = s.shift(1)
    s[(~prev.isna()) & (s != prev)] = float('nan')
    return s

def _style_tv_dark():
    mc = mpf.make_marketcolors(
        up='lime',
        down='red',
        edge='inherit',
        wick='white',
        volume='in'
    )
    return mpf.make_mpf_style(
        marketcolors=mc,
        base_mpf_style='nightclouds',
        facecolor='black',
        edgecolor='black',
        gridcolor='#333333',
        gridstyle='--',
        rc={'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white'},
        y_on_right=False
    )

def _align_channels_to_stream(ch30: pd.DataFrame, idx1m: pd.DatetimeIndex) -> pd.DataFrame:
    want = ["Value","ValueUpper","ValueLower","UpperMid","LowerMid","UpperQ","LowerQ","TrendDir"]
    ch30 = ch30.copy()
    for c in want:
        if c not in ch30.columns:
            ch30[c] = pd.NA
    out = ch30[want].reindex(idx1m.union(ch30.index)).sort_index().ffill()
    return out.reindex(idx1m)

def _has_data(s: pd.Series) -> bool:
    if s is None or len(s) == 0:
        return False
    try:
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        if arr.size == 0:
            return False
        return np.isfinite(arr).any()
    except Exception:
        return False

def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    return pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / max(length, 1)
    return series.ewm(alpha=alpha, adjust=False).mean()

def compute_supertrend(df: pd.DataFrame, atr_period: int, factor: float) -> pd.DataFrame:
    """
    Replica ta.supertrend: devuelve supertrend, direction, y series separadas up/down.
    direction=-1 → uptrend (verde), direction=+1 → downtrend (rojo).
    """
    if df is None or df.empty:
        idx = df.index if df is not None else None
        return pd.DataFrame(index=idx)

    atr = _rma(_true_range(df), max(atr_period, 1))
    hl2 = (df["High"] + df["Low"]) / 2.0
    upper_basic = hl2 + factor * atr
    lower_basic = hl2 - factor * atr

    close = df["Close"].to_numpy(dtype="float64")
    ub = upper_basic.to_numpy(dtype="float64")
    lb = lower_basic.to_numpy(dtype="float64")

    n = len(close)
    final_upper = np.copy(ub)
    final_lower = np.copy(lb)

    for i in range(1, n):
        prev_close = close[i-1]
        prev_upper = final_upper[i-1]
        prev_lower = final_lower[i-1]

        if np.isnan(prev_upper) or ub[i] < prev_upper or prev_close > prev_upper:
            final_upper[i] = ub[i]
        else:
            final_upper[i] = prev_upper

        if np.isnan(prev_lower) or lb[i] > prev_lower or prev_close < prev_lower:
            final_lower[i] = lb[i]
        else:
            final_lower[i] = prev_lower

    supertrend = np.full(n, np.nan, dtype="float64")
    direction = np.zeros(n, dtype="int8")

    for i in range(1, n):
        prev_super = supertrend[i-1]
        if np.isnan(prev_super):
            prev_super = final_upper[i-1]

        if np.isclose(prev_super, final_upper[i-1], equal_nan=False):
            if close[i] <= final_upper[i]:
                supertrend[i] = final_upper[i]
                direction[i] = 1
            else:
                supertrend[i] = final_lower[i]
                direction[i] = -1
        else:
            if close[i] >= final_lower[i]:
                supertrend[i] = final_lower[i]
                direction[i] = -1
            else:
                supertrend[i] = final_upper[i]
                direction[i] = 1

    idx = df.index
    st_series = pd.Series(supertrend, index=idx, dtype="float64")
    dir_series = pd.Series(direction, index=idx, dtype="int8")
    up_series = st_series.where(dir_series < 0)
    down_series = st_series.where(dir_series >= 0)

    return pd.DataFrame({
        "supertrend": st_series,
        "direction": dir_series,
        "supertrend_up": up_series,
        "supertrend_down": down_series,
    }, index=idx)

def main():
    # 1) Traigo mucha historia con paginado
    same_tf = STREAM_INTERVAL == CHANNEL_INTERVAL

    df_stream = fetch_klines_paginado(API_SYMBOL, STREAM_INTERVAL, PLOT_STREAM_BARS)
    ohlc_stream = df_stream[["Open","High","Low","Close","Volume"]]

    if same_tf:
        df_channel = df_stream
    else:
        df_channel = fetch_klines_paginado(API_SYMBOL, CHANNEL_INTERVAL, PLOT_CHANNEL_BARS)

    ohlc_channel = df_channel[["Open","High","Low","Close","Volume"]]

    # 2) Calculo canales en CHANNEL_INTERVAL y los extiendo al timeframe de plot
    chans_channel = compute_channels(ohlc_channel, multi=RB_MULTI, init_bar=RB_INIT_BAR)

    if same_tf:
        chans_plot = chans_channel.reindex(ohlc_stream.index).ffill()
    else:
        chans_plot = _align_channels_to_stream(chans_channel, ohlc_stream.index)

    # 3) Si no hay datos suficientes para canales, ploteo solo velas 1m
    cols_chk = ["Value","ValueUpper","ValueLower","UpperMid","LowerMid","UpperQ","LowerQ","TrendDir"]
    all_nan_cols = [c for c in cols_chk if not _has_data(chans_plot.get(c, pd.Series(dtype=float)))]
    if len(all_nan_cols) == len(cols_chk):
        print(f"[WARN] Canales/Q vacíos. Revisá datos en {CHANNEL_INTERVAL} o subí PLOT_CHANNEL_BARS.")
        mpf.plot(
            ohlc_stream, type='candle',
            style=_style_tv_dark(),
            addplot=[],
            figsize=(12,6),
            datetime_format='%Y-%m-%d %H:%M',
            title=f"{SYMBOL_DISPLAY} {STREAM_INTERVAL} — sin canales (historia insuficiente)",
            warn_too_much_data=WARN_TOO_MUCH
        )
        return

    # 4) Supertrend calculado sobre el timeframe de canales (30m) y alineado a 1m
    st_channel = compute_supertrend(ohlc_channel, atr_period=SUPER_ATR_PERIOD, factor=SUPER_FACTOR)
    if same_tf:
        st_aligned = st_channel.reindex(ohlc_stream.index).ffill()
    else:
        st_aligned = st_channel.reindex(ohlc_stream.index.union(st_channel.index)).sort_index().ffill().reindex(ohlc_stream.index)

    trend_channel = chans_channel.get("TrendDir", pd.Series(index=chans_channel.index, dtype="float64"))
    if same_tf:
        trend_aligned = trend_channel.reindex(ohlc_stream.index).ffill()
    else:
        trend_aligned = trend_channel.reindex(ohlc_stream.index.union(trend_channel.index)).sort_index().ffill().reindex(ohlc_stream.index)

    # 5) Overlays con guardas
    ap = []

    raw_val_upper = chans_plot.get('ValueUpper')
    raw_val_lower = chans_plot.get('ValueLower')
    raw_val_mid   = chans_plot.get('Value')
    raw_upper_mid = chans_plot.get('UpperMid')
    raw_lower_mid = chans_plot.get('LowerMid')

    value_upper = _linebreak_like(raw_val_upper) if raw_val_upper is not None else None
    value_lower = _linebreak_like(raw_val_lower) if raw_val_lower is not None else None
    value_mid   = _linebreak_like(raw_val_mid)   if raw_val_mid is not None else None
    upper_mid   = _linebreak_like(raw_upper_mid) if raw_upper_mid is not None else None
    lower_mid   = _linebreak_like(raw_lower_mid) if raw_lower_mid is not None else None

    if value_upper is not None and not trend_aligned.empty:
        dir1m = trend_aligned.reindex(value_upper.index).ffill()
    elif not trend_aligned.empty:
        dir1m = trend_aligned.ffill()
    else:
        base_idx = value_upper.index if value_upper is not None else ohlc_stream.index
        dir1m = pd.Series(index=base_idx, dtype="float64")

    bull_mask = (dir1m == 1.0)
    bear_mask = (dir1m == -1.0)

    if (value_upper is not None) and (value_lower is not None):
        upper_bull = value_upper.where(bull_mask)
        lower_bull = value_lower.where(bull_mask)
        upper_bear = value_upper.where(bear_mask)
        lower_bear = value_lower.where(bear_mask)
    else:
        upper_bull = lower_bull = upper_bear = lower_bear = None

    if _has_data(value_mid):
        ap.append(mpf.make_addplot(value_mid, color='#d0d0d0', width=1, linestyle='--'))
    if _has_data(upper_mid):
        ap.append(mpf.make_addplot(upper_mid, color='#9e9e9e', width=1, linestyle=':'))
    if _has_data(lower_mid):
        ap.append(mpf.make_addplot(lower_mid, color='#9e9e9e', width=1, linestyle=':'))

    if _has_data(upper_bull):
        ap.append(mpf.make_addplot(upper_bull, color=COLOR_BULL_LINE, width=1.8))
    if _has_data(lower_bull):
        ap.append(mpf.make_addplot(lower_bull, color=COLOR_BULL_LINE, width=1.8))
    if _has_data(upper_bear):
        ap.append(mpf.make_addplot(upper_bear, color=COLOR_BEAR_LINE, width=1.8))
    if _has_data(lower_bear):
        ap.append(mpf.make_addplot(lower_bear, color=COLOR_BEAR_LINE, width=1.8))

    if _has_data(value_upper) and _has_data(value_lower):
        arr_up = value_upper.to_numpy(dtype=float)
        arr_lo = value_lower.to_numpy(dtype=float)
        mask_bull = (~np.isnan(arr_up)) & (~np.isnan(arr_lo)) & (bull_mask.to_numpy(dtype=bool))
        mask_bear = (~np.isnan(arr_up)) & (~np.isnan(arr_lo)) & (bear_mask.to_numpy(dtype=bool))
        if mask_bull.any():
            ap.append(mpf.make_addplot(
                value_upper,
                color=COLOR_BULL_LINE,
                width=0,
                fill_between=dict(
                    y1=arr_up,
                    y2=arr_lo,
                    where=mask_bull,
                    alpha=CHANNEL_FILL_ALPHA,
                    color=COLOR_BULL_LINE
                )
            ))
        if mask_bear.any():
            ap.append(mpf.make_addplot(
                value_upper,
                color=COLOR_BEAR_LINE,
                width=0,
                fill_between=dict(
                    y1=arr_up,
                    y2=arr_lo,
                    where=mask_bear,
                    alpha=CHANNEL_FILL_ALPHA,
                    color=COLOR_BEAR_LINE
                )
            ))

    st_full = st_aligned.get('supertrend')
    st_dir  = st_aligned.get('direction')

    body_mid = (ohlc_stream['Open'] + ohlc_stream['Close']) / 2.0

    if _has_data(st_full) and st_dir is not None:
        up_line = st_full.where(st_dir < 0)
        dn_line = st_full.where(st_dir >= 0)
        change_mask = (st_dir != st_dir.shift(1)) & st_dir.notna() & st_dir.shift(1).notna()
        change_points = st_full.where(change_mask)

        if _has_data(up_line):
            ap.append(mpf.make_addplot(up_line, color=COLOR_BULL_LINE, width=SUPER_LINE_WIDTH))
            mask = up_line.notna().to_numpy(dtype=bool)
            if mask.any():
                ap.append(mpf.make_addplot(
                    up_line,
                    color=COLOR_BULL_LINE,
                    width=0,
                    fill_between=dict(
                        y1=up_line.to_numpy(dtype=float),
                        y2=body_mid.reindex(up_line.index).to_numpy(dtype=float),
                        where=mask,
                        alpha=SUPER_FILL_ALPHA,
                        color=COLOR_BULL_LINE
                    )
                ))

        if _has_data(dn_line):
            ap.append(mpf.make_addplot(dn_line, color=COLOR_BEAR_LINE, width=SUPER_LINE_WIDTH))
            mask = dn_line.notna().to_numpy(dtype=bool)
            if mask.any():
                ap.append(mpf.make_addplot(
                    dn_line,
                    color=COLOR_BEAR_LINE,
                    width=0,
                    fill_between=dict(
                        y1=dn_line.to_numpy(dtype=float),
                        y2=body_mid.reindex(dn_line.index).to_numpy(dtype=float),
                        where=mask,
                        alpha=SUPER_FILL_ALPHA,
                        color=COLOR_BEAR_LINE
                    )
                ))

        if _has_data(change_points):
            ap.append(mpf.make_addplot(
                change_points,
                type='scatter',
                marker='o',
                markersize=35,
                color='white'
            ))

    # 6) Plot
    mpf.plot(
        ohlc_stream, type='candle',
        style=_style_tv_dark(),
        addplot=ap, figsize=(12,6),
        datetime_format='%Y-%m-%d %H:%M',
        title=f"{SYMBOL_DISPLAY} {STREAM_INTERVAL} — canales y Supertrend (histórico paginado)",
        warn_too_much_data=WARN_TOO_MUCH
    )

if __name__ == "__main__":
    main()
