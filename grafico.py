# grafico.py
import os
import mplfinance as mpf
import pandas as pd

# IMPORTANTE: si renombraste a solo_alertas.py, usa:
# from solo_alertas import (fetch_klines, compute_channels, SYMBOL_DISPLAY, API_SYMBOL, INTERVAL, LIMIT, RB_MULTI, RB_INIT_BAR)
from soloAlertas import (
    fetch_klines, compute_channels,
    SYMBOL_DISPLAY, API_SYMBOL, INTERVAL, LIMIT,
    RB_MULTI, RB_INIT_BAR
)

# Umbral local para advertencia de mplfinance (no dependemos del módulo de datos)
WARN_TOO_MUCH = int(os.getenv("WARN_TOO_MUCH", "5000"))

def _linebreak_like(s: pd.Series) -> pd.Series:
    s = s.copy(); prev = s.shift(1)
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

def main():
    # 1) Datos
    df = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
    ohlc  = df[["Open","High","Low","Close","Volume"]]
    chans = compute_channels(ohlc, multi=RB_MULTI, init_bar=RB_INIT_BAR)

    # 2) Toques en Q-lines → puntos BLANCOS
    touch_uq = (ohlc['Low'] <= chans['UpperQ']) & (ohlc['High'] >= chans['UpperQ'])
    touch_lq = (ohlc['Low'] <= chans['LowerQ']) & (ohlc['High'] >= chans['LowerQ'])
    suq = pd.Series(float('nan'), index=ohlc.index); suq.loc[touch_uq] = chans['UpperQ'].loc[touch_uq]
    slq = pd.Series(float('nan'), index=ohlc.index); slq.loc[touch_lq] = chans['LowerQ'].loc[touch_lq]

    # 3) Overlays
    ap = [
        # Canal principal
        mpf.make_addplot(_linebreak_like(chans['ValueUpper']), color='#1dac70', width=1),
        mpf.make_addplot(_linebreak_like(chans['Value']),      color='gray',    width=1),
        mpf.make_addplot(_linebreak_like(chans['ValueLower']), color='#df3a79', width=1),
        # Mid-lines grises
        mpf.make_addplot(_linebreak_like(chans['UpperMid']),   color='gray',    width=1, alpha=0.5),
        mpf.make_addplot(_linebreak_like(chans['LowerMid']),   color='gray',    width=1, alpha=0.5),
        # Q-lines amarillas punteadas
        mpf.make_addplot(_linebreak_like(chans['UpperQ']),     color='yellow',  width=1, linestyle=':'),
        mpf.make_addplot(_linebreak_like(chans['LowerQ']),     color='yellow',  width=1, linestyle=':'),
        # Puntos BLANCOS en los toques
        mpf.make_addplot(suq, type='scatter', marker='o', markersize=40, color='white'),
        mpf.make_addplot(slq, type='scatter', marker='o', markersize=40, color='white'),
    ]

    # 4) Plot
    mpf.plot(
        ohlc, type='candle',
        style=_style_tv_dark(),
        addplot=ap, figsize=(12,6),
        datetime_format='%Y-%m-%d %H:%M',
        title=f"{SYMBOL_DISPLAY} {INTERVAL} — Q-lines amarillas + toques blancos",
        warn_too_much_data=WARN_TOO_MUCH
    )

if __name__ == "__main__":
    main()
