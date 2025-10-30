# trade_logger.py
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TRADES_PATH = os.getenv("STRAT_TRADES_CSV_PATH", "estrategia_trades.csv").strip()

TRADE_COLUMNS = [
    "EntryTime",
    "ExitTime",
    "Direction",
    "EntryPrice",
    "ExitPrice",
    "EntryReason",
    "ExitReason",
    "PnLAbs",
    "PnLPct",
    "Fees",
    "Outcome",
]


def _prepare_csv(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(path, index=False, encoding="utf-8")


def log_trade(
    *,
    direction: str,
    entry_price: float,
    exit_price: float,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_reason: str,
    exit_reason: str,
    fees: float = 0.0,
    csv_path: str | bool | None = None,
):
    if entry_price is None or exit_price is None:
        return

    path = None if csv_path is False else Path(csv_path or DEFAULT_TRADES_PATH)
    if path is not None:
        _prepare_csv(path)

    pnl_abs = exit_price - entry_price if direction == "long" else entry_price - exit_price
    pnl_pct = pnl_abs / entry_price if entry_price else np.nan
    pnl_abs -= fees
    if entry_price:
        pnl_pct = pnl_abs / entry_price
    outcome = "win" if pnl_abs > 0 else ("loss" if pnl_abs < 0 else "flat")

    data = {
        "EntryTime": entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time),
        "ExitTime": exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time),
        "Direction": direction,
        "EntryPrice": entry_price,
        "ExitPrice": exit_price,
        "EntryReason": entry_reason,
        "ExitReason": exit_reason,
        "PnLAbs": pnl_abs,
        "PnLPct": pnl_pct,
        "Fees": fees,
        "Outcome": outcome,
    }

    if path is not None:
        pd.DataFrame([data]).to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    print(
        f"[TRADE] {direction.upper()} {entry_reason} → {exit_reason} | "
        f"Entry {entry_price:.2f} Exit {exit_price:.2f} | Fees {fees:.2f} | PnL {pnl_abs:.2f} ({pnl_pct*100:.2f}%)"
    )
