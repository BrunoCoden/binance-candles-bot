# run_backtest.py
import argparse
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from binance.um_futures import UMFutures

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from .config import OUTPUT_PRESETS, resolve_profile
except ImportError:  # ejecución como script directo
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    if str(CURRENT_DIR.parent) not in sys.path:
        sys.path.append(str(CURRENT_DIR.parent))
    from config import OUTPUT_PRESETS, resolve_profile

from velas import BB_DIRECTION, BB_LENGTH, BB_MULT, API_SYMBOL, STREAM_INTERVAL, SYMBOL_DISPLAY, compute_bollinger_bands
from paginado_binance import INTERVAL_MS, fetch_klines_paginado
from trade_logger import log_trade, TRADE_COLUMNS


BACKTEST_STREAM_BARS = int(os.getenv("BACKTEST_STREAM_BARS", "5000"))
BACKTEST_CHANNEL_BARS = int(os.getenv("BACKTEST_CHANNEL_BARS", "5000"))  # legacy env, sin uso
SHOW_PLOT = os.getenv("BACKTEST_PLOT_SHOW", "false").lower() == "true"

COLUMN_ORDER = TRADE_COLUMNS


def _get_um_client() -> UMFutures:
    base_url = os.getenv("BINANCE_UM_BASE_URL", "https://fapi.binance.com")
    return UMFutures(base_url=base_url)


def _fetch_fee_rate(symbol: str) -> float:
    """
    Devuelve la comisión taker para el símbolo dado.
    Si no se puede obtener, retorna un valor por defecto de 0.0005 (0.05%).
    """
    try:
        client = _get_um_client()
        info = client.exchange_info()
        for entry in info.get("symbols", []):
            if entry.get("symbol") == symbol:
                taker = entry.get("takerCommissionRate")
                if taker is not None:
                    return float(taker)
    except Exception as exc:
        print(f"[BACKTEST][WARN] No se pudo obtener la tasa de comisión desde Binance ({exc}); se usará 0.0005.")
    return 0.0005


def _resolve_time_window(
    *,
    interval: str,
    default_bars: int,
    weeks: int | None,
    months: int | None,
    start: str | None,
    end: str | None,
) -> tuple[int, int | None, int | None, pd.Timestamp | None, pd.Timestamp | None]:
    """
    Calcula el rango temporal deseado en función de semanas/meses/start/end.
    Retorna:
        total_bars, start_ms, end_ms, start_local, end_local
    """
    if not any([weeks, months, start, end]):
        return default_bars, None, None, None, None

    interval_ms = INTERVAL_MS.get(interval)
    if not interval_ms:
        raise ValueError(f"No se reconoce el intervalo {interval} para calcular rango temporal.")

    tz_name = os.getenv("TZ", "America/Argentina/Buenos_Aires")
    local_tz = ZoneInfo(tz_name)

    end_ts_utc: pd.Timestamp
    if end:
        end_ts_utc = pd.Timestamp(end)
    else:
        end_ts_utc = pd.Timestamp.utcnow()
    if end_ts_utc.tzinfo is None:
        end_ts_utc = end_ts_utc.tz_localize("UTC")
    else:
        end_ts_utc = end_ts_utc.tz_convert("UTC")

    if start:
        start_ts_utc = pd.Timestamp(start)
        if start_ts_utc.tzinfo is None:
            start_ts_utc = start_ts_utc.tz_localize("UTC")
        else:
            start_ts_utc = start_ts_utc.tz_convert("UTC")
    else:
        start_ts_utc = end_ts_utc
        if weeks:
            start_ts_utc = start_ts_utc - pd.to_timedelta(int(weeks), unit="W")
        elif months:
            start_ts_utc = start_ts_utc - pd.DateOffset(months=int(months))
        else:
            # si se especificó solo end
            start_ts_utc = start_ts_utc - pd.to_timedelta(default_bars * interval_ms, unit="ms")

    if start_ts_utc >= end_ts_utc:
        raise ValueError("El inicio del rango debe ser anterior al fin.")

    delta_ms = (end_ts_utc - start_ts_utc) / pd.Timedelta(milliseconds=1)
    total_bars = int(math.ceil(delta_ms / interval_ms)) + 2  # margen adicional
    total_bars = max(total_bars, 2)

    start_ms = int(start_ts_utc.timestamp() * 1000)
    end_ms = int(end_ts_utc.timestamp() * 1000)

    start_local = start_ts_utc.tz_convert(local_tz)
    end_local = end_ts_utc.tz_convert(local_tz)

    return total_bars, start_ms, end_ms, start_local, end_local


def _prepare_data(total_bars: int, *, start_ms: int | None, end_ms: int | None):
    df_stream = fetch_klines_paginado(
        API_SYMBOL,
        STREAM_INTERVAL,
        total_bars,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    if df_stream.empty:
        raise RuntimeError("Datos insuficientes para backtest.")

    ohlc_stream = df_stream[["Open", "High", "Low", "Close", "Volume"]]
    bb = compute_bollinger_bands(ohlc_stream, BB_LENGTH, BB_MULT).reindex(ohlc_stream.index).ffill()
    return ohlc_stream, bb


def _generate_signal(row_idx: int, ohlc: pd.DataFrame, bb: pd.DataFrame):
    signals = []
    if row_idx == 0:
        return signals

    upper = bb.get("upper")
    lower = bb.get("lower")
    basis = bb.get("basis")
    close = ohlc["Close"]

    if upper is None or lower is None:
        return signals

    vals = [
        close.iloc[row_idx],
        close.iloc[row_idx - 1],
        upper.iloc[row_idx],
        upper.iloc[row_idx - 1],
        lower.iloc[row_idx],
        lower.iloc[row_idx - 1],
    ]

    if any(np.isnan(float(val)) for val in vals):
        return signals

    close_now = float(vals[0])
    close_prev = float(vals[1])
    upper_now = float(vals[2])
    upper_prev = float(vals[3])
    lower_now = float(vals[4])
    lower_prev = float(vals[5])

    crossed_lower = close_prev <= lower_prev and close_now > lower_now
    crossed_upper = close_prev >= upper_prev and close_now < upper_now

    ts = ohlc.index[row_idx]
    direction_filter = BB_DIRECTION
    basis_now = float(basis.iloc[row_idx]) if basis is not None and not np.isnan(basis.iloc[row_idx]) else np.nan

    if crossed_lower and direction_filter != -1:
        signals.append(
            {
                "type": "bollinger_signal",
                "direction": "long",
                "message": f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: Señal Bollinger alcista en {lower_now:.2f}",
                "price": lower_now,
                "timestamp": ts,
                "basis": basis_now,
                "reference_band": lower_now,
            }
        )
    if crossed_upper and direction_filter != 1:
        signals.append(
            {
                "type": "bollinger_signal",
                "direction": "short",
                "message": f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: Señal Bollinger bajista en {upper_now:.2f}",
                "price": upper_now,
                "timestamp": ts,
                "basis": basis_now,
                "reference_band": upper_now,
            }
        )

    return signals


def run_backtest(
    stream_bars: int,
    channel_bars: int,
    trades_path: Path,
    plot_path: Path,
    show_plot: bool,
    *,
    start_local: pd.Timestamp | None = None,
    end_local: pd.Timestamp | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
):
    ohlc, bb = _prepare_data(stream_bars, start_ms=start_ms, end_ms=end_ms)

    fee_rate = _fetch_fee_rate(API_SYMBOL)
    print(f"[BACKTEST] Fee taker estimada: {fee_rate:.6f}")

    if start_local is not None or end_local is not None:
        idx = ohlc.index
        mask = pd.Series(True, index=idx)
        if start_local is not None:
            mask &= idx >= start_local
        if end_local is not None:
            mask &= idx <= end_local
        ohlc = ohlc.loc[mask]
        bb = bb.loc[mask]

    if len(ohlc) < 2:
        raise RuntimeError("El rango temporal seleccionado devolvió menos de 2 velas; no se puede ejecutar el backtest.")
    trades = []
    position = None

    for i in range(1, len(ohlc)):
        ts = ohlc.index[i]
        price = float(ohlc["Close"].iloc[i])

        signals = _generate_signal(i, ohlc, bb)
        if not signals:
            continue

        for signal in signals:
            direction = signal["direction"]

            if position and position["direction"] == direction:
                continue

            signal_price = float(signal.get("price", price))

            reference = signal.get("reference_band")
            basis_now = signal.get("basis")

            if position:
                exit_price = reference if reference is not None else signal_price
                try:
                    exit_price = float(exit_price)
                except Exception:
                    exit_price = signal_price
                position["exit_meta"] = {"basis": basis_now}
                trades.append(_finalize_trade(position, exit_price, ts, signal["type"], fee_rate))
                position = None

            entry_price = reference if reference is not None else signal_price
            try:
                entry_price = float(entry_price)
            except Exception:
                entry_price = signal_price

            position = {
                "direction": direction,
                "entry_price": entry_price,
                "entry_time": ts,
                "entry_reason": signal["type"],
                "entry_meta": {"basis": basis_now},
            }

    if position:
        fallback_exit = position["entry_price"]
        position["exit_meta"] = {"basis": position.get("entry_meta", {}).get("basis")}
        trades.append(_finalize_trade(position, float(fallback_exit), ohlc.index[-1], "end_of_data", fee_rate))

    trades_path = Path(trades_path)
    plot_path = Path(plot_path)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    trades_df = pd.DataFrame(trades, columns=COLUMN_ORDER)
    trades_df.to_csv(trades_path, index=False)
    print(f"[BACKTEST] Guardado CSV de trades en {trades_path}")

    summary = _summarize_trades(trades_df)
    print("[BACKTEST] Resumen:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    fig = _plot_results(ohlc, trades_df, bb)
    if fig is not None:
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[BACKTEST] Gráfico guardado en {plot_path}")
        if show_plot and plt is not None:
            plt.show()
        if plt is not None:
            plt.close(fig)
    else:
        print("[BACKTEST][WARN] Matplotlib no disponible; se omitió la generación del gráfico.")


def _finalize_trade(position, exit_price, exit_time, exit_reason, fee_rate: float):
    entry_price = position["entry_price"]
    entry_time = position["entry_time"]
    direction = position["direction"]
    entry_reason = position["entry_reason"]
    entry_meta = position.get("entry_meta") or {}
    exit_meta = position.get("exit_meta") or {}

    fees = (abs(entry_price) + abs(exit_price)) * fee_rate

    log_trade(
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_reason=entry_reason,
        exit_reason=exit_reason,
        fees=fees,
        csv_path=False,  # evita escritura duplicada; se registrará manualmente más adelante
    )

    pnl_abs_raw = exit_price - entry_price if direction == "long" else entry_price - exit_price
    pnl_abs = pnl_abs_raw - fees
    pnl_pct = pnl_abs / entry_price if entry_price else np.nan
    outcome = "win" if pnl_abs > 0 else ("loss" if pnl_abs < 0 else "flat")

    return [
        entry_time.isoformat(),
        exit_time.isoformat(),
        direction,
        entry_price,
        exit_price,
        entry_reason,
        exit_reason,
        pnl_abs,
        pnl_pct,
        fees,
        outcome,
    ]


def _summarize_trades(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"Total trades": 0}
    wins = (df["Outcome"] == "win").sum()
    losses = (df["Outcome"] == "loss").sum()
    total = len(df)
    pnl_total_pct = df["PnLPct"].sum() * 100
    pnl_avg = df["PnLPct"].mean() * 100
    winrate = wins / total * 100 if total else 0
    max_drawdown = df["PnLPct"].cumsum().min() * 100
    total_fees = df.get("Fees", pd.Series(dtype=float)).sum()
    return {
        "Total trades": total,
        "Wins": wins,
        "Losses": losses,
        "Win rate %": f"{winrate:.2f}",
        "Total PnL %": f"{pnl_total_pct:.2f}",
        "Avg PnL %": f"{pnl_avg:.2f}",
        "Max Drawdown %": f"{max_drawdown:.2f}",
        "Total Fees": f"{total_fees:.2f}",
    }


def _plot_results(ohlc: pd.DataFrame, trades: pd.DataFrame, bb: pd.DataFrame):
    if plt is None:
        return None

    fig, (ax_price, ax_cum, ax_hist) = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    ax_price.plot(ohlc.index, ohlc["Close"], color="black", linewidth=1.2, label="Close")
    if bb is not None and not bb.empty:
        basis = bb.get("basis")
        upper = bb.get("upper")
        lower = bb.get("lower")
        if basis is not None:
            ax_price.plot(basis.index, basis, color="#facc15", linewidth=1.2, label="Basis")
        if upper is not None:
            ax_price.plot(upper.index, upper, color="#1dac70", linewidth=1.0, linestyle="--", label="Upper")
        if lower is not None:
            ax_price.plot(lower.index, lower, color="#dc2626", linewidth=1.0, linestyle="--", label="Lower")

    for _, trade in trades.iterrows():
        color = "green" if trade["Direction"] == "long" else "red"
        marker = "^" if trade["Direction"] == "long" else "v"
        ax_price.scatter(pd.to_datetime(trade["EntryTime"]), trade["EntryPrice"], color=color, marker=marker, s=60)
        ax_price.scatter(pd.to_datetime(trade["ExitTime"]), trade["ExitPrice"], color=color, marker="x", s=60)
    ax_price.set_title(f"{SYMBOL_DISPLAY} {STREAM_INTERVAL} — Precio y trades")
    ax_price.grid(True, linestyle="--", alpha=0.3)
    ax_price.legend(loc="upper left")

    cum_pnl = trades["PnLPct"].fillna(0).cumsum() * 100
    ax_cum.plot(pd.to_datetime(trades["ExitTime"]), cum_pnl, color="blue")
    ax_cum.set_title("PnL acumulado (%)")
    ax_cum.grid(True, linestyle="--", alpha=0.3)

    ax_hist.hist(trades["PnLPct"] * 100, bins=20, color="#888", edgecolor="black")
    ax_hist.set_title("Distribución de PnL por trade (%)")
    ax_hist.grid(True, linestyle="--", alpha=0.3)
    ax_hist.set_xlabel("% PnL")

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Backtest de estrategia Bollinger Bands.")
    parser.add_argument("--stream-bars", type=int, default=BACKTEST_STREAM_BARS)
    parser.add_argument("--profile", choices=sorted(OUTPUT_PRESETS.keys()), default=None, help="Preset de salidas (tr o historico).")
    parser.add_argument("--trades-out", type=str, default=None, help="Ruta CSV para guardar los trades.")
    parser.add_argument("--plot-out", type=str, default=None, help="Ruta PNG para guardar el gráfico.")
    parser.add_argument("--weeks", type=int, default=None, help="Cantidad de semanas hacia atrás a incluir.")
    parser.add_argument("--months", type=int, default=None, help="Cantidad de meses hacia atrás a incluir.")
    parser.add_argument("--start", type=str, default=None, help="Inicio de rango (ISO8601).")
    parser.add_argument("--end", type=str, default=None, help="Fin de rango (ISO8601).")
    parser.add_argument("--show", action="store_true", default=SHOW_PLOT)
    args = parser.parse_args()

    if args.weeks is not None and args.weeks <= 0:
        raise SystemExit("El argumento --weeks debe ser mayor que cero.")
    if args.months is not None and args.months <= 0:
        raise SystemExit("El argumento --months debe ser mayor que cero.")
    if args.weeks is not None and args.months is not None:
        raise SystemExit("Usa solo uno de --weeks o --months.")

    profile = resolve_profile(args.profile)
    print(f"[BACKTEST] Usando perfil: {profile}")
    preset_paths = OUTPUT_PRESETS[profile]

    trades_path = Path(args.trades_out) if args.trades_out else preset_paths["trades"]
    plot_path = Path(args.plot_out) if args.plot_out else preset_paths["plot"]

    total_bars, start_ms, end_ms, start_local, end_local = _resolve_time_window(
        interval=STREAM_INTERVAL,
        default_bars=args.stream_bars,
        weeks=args.weeks,
        months=args.months,
        start=args.start,
        end=args.end,
    )

    if start_local is not None or end_local is not None:
        print(
            "[BACKTEST] Rango temporal:",
            start_local.isoformat() if start_local is not None else "N/A",
            "→",
            end_local.isoformat() if end_local is not None else "N/A",
        )

    run_backtest(
        stream_bars=total_bars,
        channel_bars=BACKTEST_CHANNEL_BARS,
        trades_path=trades_path,
        plot_path=plot_path,
        show_plot=args.show,
        start_local=start_local,
        end_local=end_local,
        start_ms=start_ms,
        end_ms=end_ms,
    )


if __name__ == "__main__":
    main()
