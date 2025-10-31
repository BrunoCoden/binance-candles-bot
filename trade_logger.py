# trade_logger.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DEFAULT_TRADES_PATH = os.getenv("STRAT_TRADES_CSV_PATH", "estrategia_trades.csv").strip()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_chat_ids_raw = os.getenv("TELEGRAM_CHAT_IDS", "")
TELEGRAM_CHAT_IDS = [part.strip() for part in _chat_ids_raw.replace(";", ",").split(",") if part.strip()]
TRADE_ALERTS_ENABLED = os.getenv("TRADE_ALERTS_ENABLED", "true").lower() == "true"

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


def _send_trade_notification(text: str):
    if not TRADE_ALERTS_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in TELEGRAM_CHAT_IDS:
        payload = {
            "chat_id": chat_id,
            "text": text,
        }
        try:
            requests.post(url, json=payload, timeout=10).raise_for_status()
        except Exception as exc:
            print(f"[TRADE][WARN] No se pudo enviar alerta a Telegram ({chat_id}): {exc}")


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
    notify: bool = False,
    csv_path: str | bool | None = None,
):
    if entry_price is None or exit_price is None:
        return

    path = None if csv_path is False else Path(csv_path or DEFAULT_TRADES_PATH)
    if path is not None:
        _prepare_csv(path)

    pnl_abs = exit_price - entry_price if direction == "long" else entry_price - exit_price
    pnl_abs -= fees
    pnl_pct = pnl_abs / entry_price if entry_price else np.nan
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
    message = (
        f"[TRADE] {direction.upper()} {entry_reason} → {exit_reason} | "
        f"Entry {entry_price:.2f} Exit {exit_price:.2f} | Fees {fees:.2f} | PnL {pnl_abs:.2f} ({pnl_pct*100:.2f}%)"
    )
    print(message)
    if notify:
        try:
            ts_entry = entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time)
            ts_exit = exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time)
            tele_msg = (
                f"Operación cerrada\n"
                f"Dirección: {direction.upper()}\n"
                f"Entrada: {entry_price:.2f} ({ts_entry})\n"
                f"Salida: {exit_price:.2f} ({ts_exit})\n"
                f"Fees: {fees:.2f}\n"
                f"PnL: {pnl_abs:.2f} ({pnl_pct*100:.2f}%)"
            )
            _send_trade_notification(tele_msg)
        except Exception as exc:
            print(f"[TRADE][WARN] Error enviando alerta de trade: {exc}")


def send_trade_notification(text: str):
    _send_trade_notification(text)
