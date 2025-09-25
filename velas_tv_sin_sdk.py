# velas_TV_sin_sdk.py
# ---------------------------------------------------------
# Range Breakout + Alertas Telegram
# - Dibuja el gráfico UNA SOLA VEZ al inicio (opcional)
# - Luego actualiza SOLO una tabla CSV con la info nueva
# - Redibuja el gráfico "a pedido" si existe un archivo flag
#   (por defecto: refresh.flag)
# ---------------------------------------------------------

import os
import json
import time
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo
from pathlib import Path
from datetime import datetime

# ================== Dependencias externas ==================
try:
    import requests
    from binance.um_futures import UMFutures
except Exception as e:
    print("Faltan dependencias:", e)
    print("Instalá: pip install binance-futures-connector mplfinance pandas requests python-dateutil tzdata")
    raise

# ================== Config por entorno ==================
SYMBOL_DISPLAY = os.getenv("SYMBOL", "ETHUSDT.P")
API_SYMBOL     = SYMBOL_DISPLAY.replace(".P", "")
INTERVAL       = os.getenv("INTERVAL", "1m")
LIMIT          = int(os.getenv("LIMIT", "1200"))
TZ_NAME        = os.getenv("TZ", "America/Argentina/Buenos_Aires")

# Indicador
RB_MULTI        = float(os.getenv("RB_MULTI", "4.0"))
RB_LB           = int(os.getenv("RB_LB", "10"))
RB_FILTER_TREND = os.getenv("RB_FILTER_TREND", "false").lower() == "true"

# Telegram
TG_ENABLE   = os.getenv("TG_ENABLE", "1") == "1"
TG_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "").strip()
STATE_FILE  = Path(os.getenv("ALERT_STATE_FILE", "alert_state.json"))

# Loop y salidas
SLEEP_SEC      = int(os.getenv("SLEEP_SEC", "10"))     # cada cuántos segundos iterar
WINDOW_BARS    = int(os.getenv("WINDOW_BARS", "400"))  # barras visibles si graficamos
WARMUP_BARS    = int(os.getenv("WARMUP_BARS", "500"))  # historial extra para ATR/SMA
SHOW_WINDOW    = os.getenv("SHOW", "1") != "0"         # graficar al inicio sí/no
SAVEFIG_PATH   = os.getenv("SAVEFIG", "").strip()      # guardar PNG tras dibujar
STREAM_CSV     = Path(os.getenv("STREAM_CSV", "stream_table.csv"))  # tabla de salida
REFRESH_FLAG   = Path(os.getenv("REFRESH_FLAG", "refresh.flag"))    # archivo para forzar redibujo

# ================== Binance ==================
def get_client():
    base_url = os.getenv("BINANCE_UM_BASE_URL", "https://fapi.binance.com")
    return UMFutures(base_url=base_url)

def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    data = get_client().klines(symbol=symbol, interval=interval, limit=limit)
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

# ================== Indicador Range Breakout ==================
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
            lb_ok  = (lb == 0 or (i - lb >= 0 and lows[i - lb] > lmid[i]))
            sell_cross = (highs[i-1] >= umid[i]) and (highs[i] < umid[i])
            lb_ok2 = (lb == 0 or (i - lb >= 0 and highs[i - lb] < umid[i]))

            buy_  = buy_cross  and lb_ok
            sell_ = sell_cross and lb_ok2

            if filter_trend:
                buy_  = buy_ and t
                sell_ = sell_ and (not t)

            plot_buy[i]  = buy_
            plot_sell[i] = sell_

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

# ================== Plot helpers ==================
def _linebreak_like(series: pd.Series) -> pd.Series:
    s = series.copy()
    prev = s.shift(1)
    s[(~prev.isna()) & (s != prev)] = np.nan
    return s

def _style_tv():
    mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in')
    return mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)

def plot_once(df: pd.DataFrame, indi: dict):
    """
    Dibuja UNA vez. Reindexa todas las series del indicador al índice de df
    para que x e y tengan la misma longitud. Si no, mplfinance explota.
    """
    # Reindexar TODAS las series del indicador al índice del df a graficar
    vu = _linebreak_like(indi['value_upper'].reindex(df.index))
    vv = _linebreak_like(indi['value'].reindex(df.index))
    vl = _linebreak_like(indi['value_lower'].reindex(df.index))
    um = _linebreak_like(indi['upper_mid'].reindex(df.index))
    lm = _linebreak_like(indi['lower_mid'].reindex(df.index))

    ap = [
        mpf.make_addplot(vu, color='#1dac70', width=1),
        mpf.make_addplot(vv, color='gray',    width=1),
        mpf.make_addplot(vl, color='#df3a79', width=1),
        mpf.make_addplot(um, color='gray',    width=1, alpha=0.5),
        mpf.make_addplot(lm, color='gray',    width=1, alpha=0.5),
    ]

    # Marcadores: construir series alineadas a df.index
    def _series_from_points(index, points, cap=120):
        s = pd.Series(np.nan, index=index)
        for ts, price in points[-cap:]:
            # Posición más cercana dentro de df.index
            pos = s.index.get_indexer([ts], method='nearest')[0]
            s.iloc[pos] = price
        return s

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
    ax = axeslist[0]
    ax.set_title(f"{SYMBOL_DISPLAY} {INTERVAL}  (plot inicial)")

    if SAVEFIG_PATH:
        fig.savefig(SAVEFIG_PATH, dpi=130)
        print(f"[OK] Gráfico guardado en: {SAVEFIG_PATH}")
    if os.getenv("KEEP_OPEN", "0") == "1":
        plt.show()  # bloqueante si querés dejarlo abierto
    else:
        try:
            import matplotlib.pyplot as _plt
            _plt.close(fig)
        except Exception:
            pass

# ================== Telegram ==================
def _load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def send_telegram_message(text: str):
    if not TG_ENABLE:
        return
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"[WARN] Telegram devolvió {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[WARN] Error enviando Telegram: {e}")

def notify_signals(df: pd.DataFrame, indi: dict):
    if len(df) < 2:
        return
    ts_last_closed = df.index[-2]
    state = _load_state()
    last_buy_ts  = state.get("last_buy_ts")
    last_sell_ts = state.get("last_sell_ts")

    if len(indi['arrows_buy']):
        ts_buy, price_buy = indi['arrows_buy'][-1]
        if ts_buy == ts_last_closed and (last_buy_ts != ts_buy.isoformat()):
            msg = f"✅ <b>COMPRA ▲</b>\nSímbolo: <b>{SYMBOL_DISPLAY}</b>\nTF: <b>{INTERVAL}</b>\nHora: <b>{ts_buy.strftime('%Y-%m-%d %H:%M')}</b>\nPrecio aprox: <b>{price_buy}</b>"
            send_telegram_message(msg)
            state["last_buy_ts"] = ts_buy.isoformat()

    if len(indi['arrows_sell']):
        ts_sell, price_sell = indi['arrows_sell'][-1]
        if ts_sell == ts_last_closed and (last_sell_ts != ts_sell.isoformat()):
            msg = f"❌ <b>VENTA ▼</b>\nSímbolo: <b>{SYMBOL_DISPLAY}</b>\nTF: <b>{INTERVAL}</b>\nHora: <b>{ts_sell.strftime('%Y-%m-%d %H:%M')}</b>\nPrecio aprox: <b>{price_sell}</b>"
            send_telegram_message(msg)
            state["last_sell_ts"] = ts_sell.isoformat()

    _save_state(state)

# ================== Tabla incremental ==================
def append_stream_row(csv_path: Path, row: dict, keep_last: int = 5000):
    """Escribe una fila al CSV y mantiene el tamaño acotado."""
    header_needed = not csv_path.exists()
    try:
        df_row = pd.DataFrame([row])
        df_row.to_csv(csv_path, mode='a', index=False, header=header_needed, encoding='utf-8')
    except PermissionError:
        # archivo abierto en Excel, te lo reservaste para mirar nada
        pass
    # recorte ocasional
    try:
        if csv_path.stat().st_size > 5_000_000:  # 5 MB
            df = pd.read_csv(csv_path)
            if len(df) > keep_last:
                df.tail(keep_last).to_csv(csv_path, index=False)
    except Exception:
        pass

# ================== Main loop ==================
def main():
    print(f"[INIT] Bajando {LIMIT} velas de {API_SYMBOL} {INTERVAL} (USDⓈ-M)")
    full = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
    end = len(full)
    start = max(0, end - (WARMUP_BARS + WINDOW_BARS))
    warm = full.iloc[start:end].copy()
    indi = compute_range_breakout(warm, multi=RB_MULTI, lb=RB_LB, filter_trend=RB_FILTER_TREND)

    # Plot inicial UNA VEZ (opcional)
    if SHOW_WINDOW:
        dfw = warm.iloc[max(0, len(warm)-WINDOW_BARS):].copy()
        plot_once(dfw, indi)

    last_closed_ts = None

    while True:
        try:
            full = fetch_klines(API_SYMBOL, INTERVAL, LIMIT)
            if len(full) < 2:
                time.sleep(SLEEP_SEC); continue

            ts_closed = full.index[-2]
            if last_closed_ts is not None and ts_closed == last_closed_ts:
                # nada nuevo; check si pediste redibujo
                if REFRESH_FLAG.exists() and SHOW_WINDOW:
                    end = len(full)
                    start = max(0, end - (WARMUP_BARS + WINDOW_BARS))
                    warm = full.iloc[start:end].copy()
                    indi = compute_range_breakout(warm, multi=RB_MULTI, lb=RB_LB, filter_trend=RB_FILTER_TREND)
                    dfw = warm.iloc[max(0, len(warm)-WINDOW_BARS):].copy()
                    plot_once(dfw, indi)
                    try: REFRESH_FLAG.unlink()
                    except Exception: pass
                time.sleep(SLEEP_SEC); continue

            # nueva vela cerrada
            last_closed_ts = ts_closed

            # recortar para performance
            end = len(full)
            start = max(0, end - (WARMUP_BARS + WINDOW_BARS))
            warm = full.iloc[start:end].copy()
            indi = compute_range_breakout(warm, multi=RB_MULTI, lb=RB_LB, filter_trend=RB_FILTER_TREND)

            # alertas
            notify_signals(warm, indi)

            # fila para la tabla CSV (penúltima vela)
            ts = warm.index[-2]
            row = {
                "time": ts.strftime("%Y-%m-%d %H:%M"),
                "symbol": SYMBOL_DISPLAY,
                "tf": INTERVAL,
                "close": float(warm['Close'].iloc[-2]),
                "value": float(indi['value'].reindex(warm.index).iloc[-2]) if not np.isnan(indi['value'].reindex(warm.index).iloc[-2]) else np.nan,
                "upper": float(indi['value_upper'].reindex(warm.index).iloc[-2]) if not np.isnan(indi['value_upper'].reindex(warm.index).iloc[-2]) else np.nan,
                "lower": float(indi['value_lower'].reindex(warm.index).iloc[-2]) if not np.isnan(indi['value_lower'].reindex(warm.index).iloc[-2]) else np.nan,
                "upper_mid": float(indi['upper_mid'].reindex(warm.index).iloc[-2]) if not np.isnan(indi['upper_mid'].reindex(warm.index).iloc[-2]) else np.nan,
                "lower_mid": float(indi['lower_mid'].reindex(warm.index).iloc[-2]) if not np.isnan(indi['lower_mid'].reindex(warm.index).iloc[-2]) else np.nan,
                "buy_signal":  1 if (len(indi['arrows_buy'])  and indi['arrows_buy'][-1][0]  == ts) else 0,
                "sell_signal": 1 if (len(indi['arrows_sell']) and indi['arrows_sell'][-1][0] == ts) else 0,
            }
            append_stream_row(STREAM_CSV, row)

            # redibujo a pedido
            if REFRESH_FLAG.exists() and SHOW_WINDOW:
                dfw = warm.iloc[max(0, len(warm)-WINDOW_BARS):].copy()
                plot_once(dfw, indi)
                try: REFRESH_FLAG.unlink()
                except Exception: pass

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(SLEEP_SEC)

# ================== Entrypoint ==================
if __name__ == "__main__":
    main()
