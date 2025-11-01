"""
Actualiza las salidas del backtest (perfil TR) en tiempo real a partir de señales.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import OUTPUT_PRESETS, resolve_profile
from .run_backtest import _finalize_trade, _fetch_fee_rate, BACKTEST_STREAM_BARS
from velas import API_SYMBOL, STREAM_INTERVAL, compute_bollinger_bands, BB_LENGTH, BB_MULT
from trade_logger import TRADE_COLUMNS
from paginado_binance import fetch_klines_paginado, INTERVAL_MS


BACKTEST_REALTIME_ENABLED = os.getenv("BACKTEST_REALTIME_ENABLED", "true").lower() == "true"
REALTIME_PROFILE = os.getenv("BACKTEST_REALTIME_PROFILE", "tr").lower()
STATE_PATH_ENV = os.getenv("BACKTEST_REALTIME_STATE_PATH", "")
STOP_LOSS_PCT = float(os.getenv("STRAT_STOP_LOSS_PCT", "0.055"))
TAKE_PROFIT_PCT = float(os.getenv("STRAT_TAKE_PROFIT_PCT", "0.095"))

PricePath = Path(os.getenv("ALERTS_TABLE_CSV_PATH", "alerts_stream.csv"))


def _ensure_timestamp(value: Any) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


@lru_cache(maxsize=1)
def _fee_rate() -> float:
    return _fetch_fee_rate(API_SYMBOL)


def _state_path(trades_path: Path) -> Path:
    if STATE_PATH_ENV:
        return Path(STATE_PATH_ENV)
    return trades_path.with_name("realtime_state.json")


def _load_state(trades_path: Path) -> Dict[str, Any] | None:
    path = _state_path(trades_path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _save_state(state: Dict[str, Any] | None, trades_path: Path) -> None:
    path = _state_path(trades_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if state is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    with path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)


def _append_trade_row(trades_path: Path, row: list[Any]) -> None:
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    header = not trades_path.exists()
    df = pd.DataFrame([row], columns=TRADE_COLUMNS)
    df.to_csv(trades_path, mode="a", header=header, index=False, encoding="utf-8")


def _rebuild_dashboard(profile: str, trades_path: Path) -> None:
    preset_paths = OUTPUT_PRESETS[profile]
    html_path = preset_paths["dashboard"]

    # Regenera dashboard utilizando el script existente.
    from .build_dashboard import render_dashboard

    try:
        render_dashboard(trades_path, PricePath if PricePath.exists() else None, html_path, show=False, profile=profile)
    except Exception as exc:
        print(f"[REALTIME][WARN] No se pudo regenerar el dashboard ({exc})")


def _refresh_plot(trades_path: Path) -> None:
    # Genera nuevamente el PNG estático reutilizando la lógica del backtest.
    preset_paths = OUTPUT_PRESETS[REALTIME_PROFILE]
    plot_path = preset_paths["plot"]

    try:
        total_bars = BACKTEST_STREAM_BARS
        df_stream = fetch_klines_paginado(
            API_SYMBOL,
            STREAM_INTERVAL,
            total_bars,
        )
        if df_stream.empty:
            return
        ohlc = df_stream[["Open", "High", "Low", "Close", "Volume"]].copy()
        if "CloseTimeDT" in df_stream.columns:
            ohlc["BarCloseTime"] = df_stream["CloseTimeDT"]
        else:
            offset = INTERVAL_MS.get(STREAM_INTERVAL, 0)
            ohlc["BarCloseTime"] = df_stream.index + pd.to_timedelta(offset, unit="ms")

        trades_df = pd.read_csv(trades_path, parse_dates=["EntryTime", "ExitTime"])
        if trades_df.empty:
            return

        from .run_backtest import _plot_results

        bb = compute_bollinger_bands(ohlc, BB_LENGTH, BB_MULT).reindex(ohlc.index).ffill()
        fig = _plot_results(ohlc, trades_df, bb)
        if fig is not None:
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            from matplotlib import pyplot as plt

            plt.close(fig)
    except Exception as exc:
        print(f"[REALTIME][WARN] No se pudo regenerar el gráfico ({exc})")


def _compute_risk_levels(direction: str, entry_price: float) -> tuple[float | None, float | None]:
    stop_price = None
    take_price = None

    if entry_price is None or entry_price <= 0:
        return stop_price, take_price

    if STOP_LOSS_PCT > 0:
        stop_price = entry_price * (1 - STOP_LOSS_PCT) if direction == "long" else entry_price * (1 + STOP_LOSS_PCT)
    if TAKE_PROFIT_PCT > 0:
        take_price = entry_price * (1 + TAKE_PROFIT_PCT) if direction == "long" else entry_price * (1 - TAKE_PROFIT_PCT)

    return stop_price, take_price


def process_realtime_signal(signal: dict[str, Any], *, profile: str = "tr") -> None:
    """
    Actualiza el CSV del backtest TR y el dashboard cuando llega una nueva señal.
    """
    if not BACKTEST_REALTIME_ENABLED:
        return
    resolved_profile = resolve_profile(profile)
    if resolved_profile != REALTIME_PROFILE:
        return

    preset_paths = OUTPUT_PRESETS[resolved_profile]
    trades_path = preset_paths["trades"]
    trades_path.parent.mkdir(parents=True, exist_ok=True)

    state = _load_state(trades_path)

    direction = signal.get("direction")
    if not direction:
        return

    ts_raw = signal.get("timestamp")
    signal_ts = _ensure_timestamp(ts_raw)
    reference_band = signal.get("reference_band")
    close_raw = reference_band if reference_band is not None else signal.get("price")
    try:
        trade_price = float(close_raw)
    except Exception:
        trade_price = float(signal.get("price", 0.0))

    basis_now = signal.get("basis")
    signal_type = signal.get("type", "unknown_signal")

    if state and state.get("direction") == direction:
        # Señal en la misma dirección que la posición abierta; se ignora (misma lógica del backtest).
        return

    fee_rate = _fee_rate()

    if state:
        try:
            position = {
                "direction": state["direction"],
                "entry_price": float(state["entry_price"]),
                "entry_time": _ensure_timestamp(state["entry_time"]),
                "entry_reason": state.get("entry_reason", "signal"),
                "entry_meta": state.get("entry_meta") or {},
            }
            position["exit_meta"] = {
                "basis": basis_now,
                "reference_band": reference_band,
                "stop_price": state.get("stop_price"),
                "take_price": state.get("take_price"),
            }
            exit_price = float(reference_band) if reference_band is not None else trade_price
            row = _finalize_trade(position, exit_price, signal_ts, signal_type, fee_rate)
            _append_trade_row(trades_path, row)
        except Exception as exc:
            print(f"[REALTIME][WARN] No se pudo cerrar la posición previa ({exc})")

    stop_price, take_price = _compute_risk_levels(direction, trade_price)

    new_state = {
        "direction": direction,
        "entry_price": trade_price,
        "entry_time": signal_ts.isoformat(),
        "entry_reason": signal_type,
        "entry_meta": {
            "basis": basis_now,
            "reference_band": reference_band,
        },
    }
    if stop_price is not None:
        new_state["stop_price"] = float(stop_price)
    if take_price is not None:
        new_state["take_price"] = float(take_price)
    _save_state(new_state, trades_path)

    if trades_path.exists():
        try:
            _rebuild_dashboard(resolved_profile, trades_path)
            _refresh_plot(trades_path)
        except Exception:
            pass


def evaluate_realtime_risk(ohlc_stream: pd.DataFrame, *, profile: str = "tr") -> None:
    """
    Verifica si la posición abierta alcanzó SL/TP usando las velas disponibles.
    """
    if not BACKTEST_REALTIME_ENABLED or ohlc_stream.empty:
        return

    resolved_profile = resolve_profile(profile)
    if resolved_profile != REALTIME_PROFILE:
        return

    trades_path = OUTPUT_PRESETS[resolved_profile]["trades"]
    state = _load_state(trades_path)
    if not state:
        return

    direction = state.get("direction")
    entry_price = float(state.get("entry_price", 0.0))
    stop_price = state.get("stop_price")
    take_price = state.get("take_price")
    entry_time = _ensure_timestamp(state.get("entry_time"))

    if direction not in {"long", "short"} or entry_price <= 0:
        return

    position_data = ohlc_stream.loc[entry_time:]
    if position_data.empty:
        return

    fee_rate = _fee_rate()

    for idx, row in position_data.iterrows():
        bar_high = float(row["High"])
        bar_low = float(row["Low"])
        ts_close = row.get("BarCloseTime", idx)

        exit_price = None
        exit_reason = None

        if direction == "long":
            if stop_price is not None and bar_low <= stop_price:
                exit_price = float(stop_price)
                exit_reason = "stop_loss"
            elif take_price is not None and bar_high >= take_price:
                exit_price = float(take_price)
                exit_reason = "take_profit"
        else:
            if stop_price is not None and bar_high >= stop_price:
                exit_price = float(stop_price)
                exit_reason = "stop_loss"
            elif take_price is not None and bar_low <= take_price:
                exit_price = float(take_price)
                exit_reason = "take_profit"

        if exit_reason:
            position = {
                "direction": direction,
                "entry_price": entry_price,
                "entry_time": _ensure_timestamp(state["entry_time"]),
                "entry_reason": state.get("entry_reason", "signal"),
                "entry_meta": state.get("entry_meta") or {},
                "stop_price": stop_price,
                "take_price": take_price,
            }
            position["exit_meta"] = {
                "stop_price": stop_price,
                "take_price": take_price,
            }
            exit_ts = ts_close if isinstance(ts_close, pd.Timestamp) else _ensure_timestamp(ts_close)
            row_data = _finalize_trade(position, exit_price, exit_ts, exit_reason, fee_rate)
            _append_trade_row(trades_path, row_data)
            _save_state(None, trades_path)
            if trades_path.exists():
                try:
                    _rebuild_dashboard(resolved_profile, trades_path)
                    _refresh_plot(trades_path)
                except Exception:
                    pass
            break
