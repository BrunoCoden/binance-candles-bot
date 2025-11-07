# watcher_alertas.py
import os
import time
from datetime import datetime, timezone, timedelta

from alerts import generate_alerts, send_alerts, format_alert_message
from trade_logger import send_trade_notification, format_timestamp
from velas import SYMBOL_DISPLAY, STREAM_INTERVAL
from trading.accounts.manager import AccountManager
from trading.orders.executor import OrderExecutor
from trading.orders.models import OrderRequest, OrderSide, OrderType

TRADING_ENABLED = os.getenv("WATCHER_ENABLE_TRADING", "false").lower() == "true"
TRADING_ACCOUNTS_FILE = os.getenv("WATCHER_ACCOUNTS_FILE", "trading/accounts/sample_accounts.yaml")
TRADING_USER_ID = os.getenv("WATCHER_TRADING_USER", "default")
TRADING_EXCHANGE = os.getenv("WATCHER_TRADING_EXCHANGE", "binance")
TRADING_DEFAULT_QTY = os.getenv("WATCHER_TRADING_DEFAULT_QTY", "0.01")
TRADING_DRY_RUN = os.getenv("WATCHER_TRADING_DRY_RUN", "true").lower() != "false"
TRADING_MIN_PRICE = float(os.getenv("WATCHER_TRADING_MIN_PRICE", "0"))

_executor: OrderExecutor | None = None


def _resolve_executor() -> OrderExecutor | None:
    global _executor
    if not TRADING_ENABLED:
        return None
    if _executor is not None:
        return _executor
    try:
        manager = AccountManager.from_file(TRADING_ACCOUNTS_FILE)
    except Exception as exc:
        print(f"[WATCHER][WARN] No se pudo inicializar AccountManager ({exc}); modo trading deshabilitado.")
        return None
    _executor = OrderExecutor(manager)
    return _executor


def _direction_to_side(direction: str | None) -> OrderSide:
    mapping = {
        "long": OrderSide.BUY,
        "short": OrderSide.SELL,
    }
    key = (direction or "").lower()
    if key not in mapping:
        raise ValueError(f"Dirección inválida para operar: {direction}")
    return mapping[key]


def _resolve_quantity(event: dict) -> float:
    qty_raw = event.get("quantity") or TRADING_DEFAULT_QTY
    qty = float(str(qty_raw).replace(",", "."))
    if qty <= 0:
        raise ValueError("quantity debe ser > 0")
    return qty


def _price_from_event(event: dict) -> float | None:
    for key in ("reference_band", "price", "entry_price"):
        val = event.get(key)
        if val is None:
            continue
        try:
            price = float(val)
            if price > 0:
                return price
        except Exception:
            continue
    close_price = event.get("close_price")
    if close_price is not None:
        try:
            price = float(close_price)
            if price > 0:
                return price
        except Exception:
            pass
    return None


def _submit_trade(event: dict) -> None:
    executor = _resolve_executor()
    if executor is None:
        return
    try:
        side = _direction_to_side(event.get("direction"))
    except Exception as exc:
        print(f"[WATCHER][WARN] No se pudo determinar dirección para trading: {exc}")
        return
    price = _price_from_event(event)
    if price is None or price <= 0:
        print("[WATCHER][WARN] Evento sin precio de referencia, se omite trading.")
        return
    if TRADING_MIN_PRICE > 0 and price < TRADING_MIN_PRICE:
        print(f"[WATCHER][INFO] Precio {price:.2f} < mínimo configurado ({TRADING_MIN_PRICE}); no se opera.")
        return
    try:
        quantity = _resolve_quantity(event)
    except Exception as exc:
        print(f"[WATCHER][WARN] Cantidad inválida para trading ({exc})")
        return
    order = OrderRequest(
        symbol=event.get("symbol") or SYMBOL_DISPLAY.replace(".P", ""),
        side=side,
        type=OrderType.MARKET,
        quantity=quantity,
        extra_params={
            "source_event": event.get("type", "unknown"),
            "event_timestamp": str(event.get("timestamp")),
        },
    )
    try:
        response = executor.execute(TRADING_USER_ID, TRADING_EXCHANGE, order, dry_run=TRADING_DRY_RUN)
        print(f"[WATCHER][TRADE] Resultado orden: success={response.success} status={response.status} raw={response.raw}")
    except Exception as exc:
        print(f"[WATCHER][ERROR] Falló la ejecución de orden ({exc})")

POLL_SECONDS = float(os.getenv("ALERT_POLL_SECONDS", "5"))
MAX_SEEN = int(os.getenv("ALERT_MAX_SEEN", "500"))
SEND_STARTUP_TEST = os.getenv("WATCHER_STARTUP_TEST_ALERT", "true").lower() == "true"


def _notify_startup():
    if not SEND_STARTUP_TEST:
        return
    try:
        now_utc = datetime.now(timezone.utc)
        entry_time = now_utc - timedelta(minutes=45)
        ts_entry = format_timestamp(entry_time)
        ts_exit = format_timestamp(now_utc)

        opening_msg = (
            f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}\n"
            f"[PRUEBA] Apertura LONG\n"
            f"Precio: 3500.00\n"
            f"Hora: {ts_entry}\n"
            f"Motivo: ALERTA_DE_PRUEBA"
        )

        closing_msg = (
            f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}\n"
            f"[PRUEBA] Cierre LONG\n"
            f"Entrada: 3500.00 ({ts_entry})\n"
            f"Salida: 3600.00 ({ts_exit})\n"
            f"Fees: 3.50\n"
            f"Resultado: GANANCIA 96.50 (+2.76%)"
        )

        send_trade_notification(opening_msg)
        send_trade_notification(closing_msg)

        sample_alert = {
            "type": "heartbeat_test",
            "timestamp": now_utc,
            "message": (
                f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: [PRUEBA] Señal formateada\n"
                f"Entrada simulada 3500 → 3600 (+2.76%)"
            ),
            "direction": "long",
        }
        send_alerts([sample_alert])
    except Exception as exc:
        print(f"[WATCHER][WARN] No se pudo enviar la alerta de prueba: {exc}")


def main():
    seen = []
    _notify_startup()
    while True:
        try:
            events = generate_alerts()
        except Exception as exc:
            print(f"[ERROR] {exc}")
            time.sleep(POLL_SECONDS)
            continue

        new_alerts = []
        for evt in events:
            ts = evt.get("timestamp")
            key = (evt.get("type"), ts)
            if key in seen:
                continue
            seen.append(key)
            seen[:] = seen[-MAX_SEEN:]

            print(f"[ALERTA] {format_alert_message(evt)}")
            new_alerts.append(evt)

        if new_alerts:
            send_alerts(new_alerts)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
