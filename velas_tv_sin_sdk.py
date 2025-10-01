# velas_TV_sin_sdk.py
# ---------------------------------------------------------
# Velas + Range Breakout (canal + flechas)
# - SIN círculos: solo flechas ▲ (buy) y ▼ (sell)
# - Las rupturas de canal también generan flechas
# - Respeta filtro de tendencia por env RB_FILTER_TREND
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
        # Inicializa canal cuando hay historial suficiente (estilo Pine)
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
        cross_up   = crossover(lows[i], vup[i], low_prev)   # ruptura al techo
        cross_down = crossunder(highs[i], vlo[i], high_prev) # ruptura al piso

        # Conteo de barras fuera del canal
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

        # Actualiza tendencia a partir de la ruptura
        if cross_up:
            t = True
        if cross_down:
            t = False

        trend[i] = t
        chage = not channel_changed  # true si el canal NO cambió en esta vela

        # ====== FLECHAS POR CRUCE DE MEDIAS (en i-1, estilo Pine) ======
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

        # ====== FLECHAS POR RUPTURA DE CANAL (en i) ======
        if cross_up and (t if filter_trend else True):
            plot_buy[i] = True
            arrows_buy.append((idx[i], lows[i]))
        if cross_down and ((not t) if filter_trend else True):
            plot_sell[i] = True
            arrows_sell.append((idx[i], highs[i]))

        # dedupe exacto
        if len(arrows_buy) >= 2 and arrows_buy[-1] == arrows_buy[-2]:
            arrows_buy.pop()
        if len(arrows_sell) >= 2 and arrows_sell[-1] == arrows_sell[-2]:
            arrows_sell.pop()

    return {
        'value': pd.Series(value, index=df.index),
        'value_upper': pd.Series(vup, index=df.index),
        'value_lower': pd.Series(vlo, index=df.index),
        'upper_mid': pd.Series(umid, index=df.index),
        'lower_mid': pd.Series(lmid, index=df.index),
        'arrows_buy': arrows_buy,
        'arrows_sell': arrows_sell,
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

    # Solo flechas (sin círculos)
    buy_s  = _series_from_points(df.index, indi['arrows_buy'])
    sell_s = _series_from_points(df.index, indi['arrows_sell'])

    ap += [
        mpf.make_addplot(buy_s,  type='scatter', marker='^', markersize=60, color='#1dac70'),
        mpf.make_addplot(sell_s, type='scatter', marker='v', markersize=60, color='#df3a79'),
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

    # Debug rápido: cantidades de flechas
    print(f"[DEBUG] Flechas BUY: {len(indi['arrows_buy'])} | Flechas SELL: {len(indi['arrows_sell'])}")

    fig, ax = plot_range_breakout(df, indi)

    if SAVEFIG_PATH:
        fig.savefig(SAVEFIG_PATH, dpi=130)
        print(f"[OK] Gráfico guardado en: {SAVEFIG_PATH}")
    if SHOW_WINDOW:
        mpf.show()

if __name__ == "__main__":
    main()
