# velas_TV_sin_sdk.py
# ---------------------------------------------------------
# Velas + Range Breakout (canal + flechas + rupturas)
# Círculos de ruptura se dibujan sólidos (sin hueco).
# ---------------------------------------------------------

import os
import numpy as np
import pandas as pd
import mplfinance as mpf
from zoneinfo import ZoneInfo

try:
    from binance.um_futures import UMFutures
except Exception as e:
    print("ERROR: Falta el conector de Futuros de Binance.")
    print("Instalá con:  pip install binance-futures-connector")
    raise

# ================== Config ==================
SYMBOL_DISPLAY = os.getenv("SYMBOL", "ETHUSDT.P")
API_SYMBOL     = SYMBOL_DISPLAY.replace(".P", "")
INTERVAL       = os.getenv("INTERVAL", "30m")
LIMIT          = int(os.getenv("LIMIT", "1200"))
TZ_NAME        = os.getenv("TZ", "America/Argentina/Buenos_Aires")

RB_MULTI        = float(os.getenv("RB_MULTI", "4.0"))
RB_LB           = int(os.getenv("RB_LB", "10"))
RB_FILTER_TREND = os.getenv("RB_FILTER_TREND", "false").lower() == "true"

SAVEFIG_PATH   = os.getenv("SAVEFIG", "").strip()
SHOW_WINDOW    = os.getenv("SHOW", "1") != "0"

# ================== Datos ==================
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
    df = df.set_index("Date").sort_index()
    df.index = df.index.tz_convert(ZoneInfo(TZ_NAME))
    return df[["Open","High","Low","Close","Volume"]]

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
    breaks_up, breaks_dn = [], []

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
        cross_up   = crossover(lows[i], vup[i], low_prev)
        cross_down = crossunder(highs[i], vlo[i], high_prev)

        if not np.isnan(vup[i]) and not np.isnan(vlo[i]):
            if lows[i] > vup[i] or highs[i] < vlo[i]:
                count += 1

        channel_changed = False
        if (cross_up or cross_down or count == 100):
            count = 0
            value[i] = hl2[i]; vup[i] = hl2[i] + w[i]; vlo[i] = hl2[i] - w[i]
            umid[i] = (value[i] + vup[i]) / 2.0; lmid[i] = (value[i] + vlo[i]) / 2.0
            channel_changed = True

        if cross_up:
            t = True; breaks_up.append((idx[i-1], highs[i-1]))
        if cross_down:
            t = False; breaks_dn.append((idx[i-1], lows[i-1]))

        trend[i] = t
        chage = not channel_changed

        if chage and not np.isnan(lmid[i]) and not np.isnan(umid[i]):
            buy_cross = (lows[i-1] <= lmid[i]) and (lows[i] > lmid[i])
            lb_ok = (lb == 0 or lows[i - lb] > lmid[i]) if i - lb >= 0 else True
            sell_cross = (highs[i-1] >= umid[i]) and (highs[i] < umid[i])
            lb_ok2 = (lb == 0 or highs[i - lb] < umid[i]) if i - lb >= 0 else True

            buy_  = buy_cross  and lb_ok
            sell_ = sell_cross and lb_ok2

            plot_buy[i]  = buy_  and (t if filter_trend else True)
            plot_sell[i] = sell_ and ((not t) if filter_trend else True)

            if plot_buy[i]:  arrows_buy.append((idx[i-1], lows[i-1]))
            if plot_sell[i]: arrows_sell.append((idx[i-1], highs[i-1]))

    return {
        'value': pd.Series(value, index=df.index),
        'value_upper': pd.Series(vup, index=df.index),
        'value_lower': pd.Series(vlo, index=df.index),
        'upper_mid': pd.Series(umid, index=df.index),
        'lower_mid': pd.Series(lmid, index=df.index),
        'arrows_buy': arrows_buy,
        'arrows_sell': arrows_sell,
        'breaks_up': breaks_up,
        'breaks_dn': breaks_dn,
    }

# ================== Plot ==================
def _linebreak_like(series: pd.Series) -> pd.Series:
    s = series.copy(); prev = s.shift(1)
    s[(~prev.isna()) & (s != prev)] = np.nan
    return s

def _style_tv():
    mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in')
    return mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)

def _series_from_points(index, points):
    s = pd.Series(np.nan, index=index)
    for ts, price in points:
        pos = s.index.get_indexer([ts], method='nearest')[0]
        s.iloc[pos] = price
    return s

def plot_range_breakout(df, indi):
    ap = [
        mpf.make_addplot(_linebreak_like(indi['value_upper']), color='#1dac70', width=1),
        mpf.make_addplot(_linebreak_like(indi['value']),       color='gray',    width=1),
        mpf.make_addplot(_linebreak_like(indi['value_lower']), color='#df3a79', width=1),
        mpf.make_addplot(_linebreak_like(indi['upper_mid']),   color='gray',    width=1, alpha=0.5),
        mpf.make_addplot(_linebreak_like(indi['lower_mid']),   color='gray',    width=1, alpha=0.5),
    ]

    buy_s  = _series_from_points(df.index, indi['arrows_buy'])
    sell_s = _series_from_points(df.index, indi['arrows_sell'])
    up_s   = _series_from_points(df.index, indi['breaks_up'])
    dn_s   = _series_from_points(df.index, indi['breaks_dn'])

    ap += [
        mpf.make_addplot(buy_s,  type='scatter', marker='^', markersize=60, color='#1dac70'),
        mpf.make_addplot(sell_s, type='scatter', marker='v', markersize=60, color='#df3a79'),
        mpf.make_addplot(up_s,   type='scatter', marker='o', markersize=30, color='#1dac70'),
        mpf.make_addplot(dn_s,   type='scatter', marker='o', markersize=30, color='#df3a79'),
    ]

    fig, axeslist = mpf.plot(
        df,
        type='candle',
        style=_style_tv(),
        addplot=ap,
        returnfig=True,
        figsize=(12, 6),
        datetime_format='%Y-%m-%d %H:%M',
        warn_too_much_data=len(df)+1
    )

    return fig, axeslist[0]

# ================== Main ==================
def main():
    print(f"[INFO] Bajando {LIMIT} velas de {API_SYMBOL} {INTERVAL} (USDⓈ-M)")
    df = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
    indi = compute_range_breakout(df, multi=RB_MULTI, lb=RB_LB, filter_trend=RB_FILTER_TREND)

    fig, ax = plot_range_breakout(df, indi)

    if SAVEFIG_PATH:
        fig.savefig(SAVEFIG_PATH, dpi=130)
        print(f"[OK] Gráfico guardado en: {SAVEFIG_PATH}")
    if SHOW_WINDOW:
        mpf.show()

if __name__ == "__main__":
    main()
