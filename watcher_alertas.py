# watcher_alertas.py
import os
import math
from pathlib import Path
import time
from datetime import datetime, timezone, timedelta

from binance.um_futures import UMFutures

from alerts import generate_alerts, send_alerts, format_alert_message
from trade_logger import send_trade_notification, format_timestamp
from velas import SYMBOL_DISPLAY, STREAM_INTERVAL
from trading.accounts.manager import AccountManager
from trading.accounts.models import ExchangeEnvironment
from trading.orders.executor import OrderExecutor
from trading.orders.models import OrderRequest, OrderSide, OrderType, TimeInForce

TRADING_ENABLED = os.getenv("WATCHER_ENABLE_TRADING", "false").lower() == "true"
TRADING_ACCOUNTS_FILE = os.getenv("WATCHER_ACCOUNTS_FILE", "trading/accounts/sample_accounts.yaml")
TRADING_USER_ID = os.getenv("WATCHER_TRADING_USER", "").strip()
TRADING_EXCHANGE = os.getenv("WATCHER_TRADING_EXCHANGE", "").strip()
TRADING_DEFAULT_QTY = os.getenv("WATCHER_TRADING_DEFAULT_QTY", "0.01")
# Si se indica, calcula cantidad a partir de un notional USDT (qty = notional / price)
TRADING_DEFAULT_NOTIONAL_USDT = float(os.getenv("WATCHER_TRADING_NOTIONAL_USDT", "0") or 0)
TRADING_DRY_RUN = os.getenv("WATCHER_TRADING_DRY_RUN", "true").lower() != "false"
TRADING_MIN_PRICE = float(os.getenv("WATCHER_TRADING_MIN_PRICE", "0"))
TRADING_MIN_NOTIONAL = float(os.getenv("WATCHER_MIN_NOTIONAL_USDT", "20"))

_executor: OrderExecutor | None = None
_account_manager: AccountManager | None = None
_last_order_direction: str | None = None
_pending_signals: list[dict] = []


def _load_manager() -> AccountManager | None:
    global _account_manager
    if not TRADING_ENABLED:
        return None
    try:
        path = Path(TRADING_ACCOUNTS_FILE)
        _account_manager = AccountManager.from_file(path)
        return _account_manager
    except Exception as exc:
        print(f"[WATCHER][WARN] No se pudo inicializar AccountManager ({exc}); modo trading deshabilitado.")
        return None


def _resolve_executor() -> OrderExecutor | None:
    global _executor
    manager = _load_manager()
    if manager is None:
        return None
    if _executor is not None:
        return _executor
    _executor = OrderExecutor(manager)
    return _executor


def _resolve_targets() -> list[tuple[str, str]]:
    """
    Devuelve lista de (user_id, exchange) habilitados.
    Si se configuró WATCHER_TRADING_USER/EXCHANGE se usa como filtro.
    """
    manager = _load_manager()
    if manager is None:
        return []

    user_filter = TRADING_USER_ID.lower()
    if user_filter in {"", "default"}:
        user_filter = None
    exchange_filter = TRADING_EXCHANGE.lower() if TRADING_EXCHANGE else None

    targets: list[tuple[str, str]] = []
    for account in manager.list_accounts():
        if not account.enabled:
            continue
        if user_filter and account.user_id.lower() != user_filter:
            continue
        for ex_name, cred in account.exchanges.items():
            if exchange_filter and ex_name.lower() != exchange_filter:
                continue
            targets.append((account.user_id, ex_name))
    return targets


def _direction_to_side(direction: str | None) -> OrderSide:
    mapping = {
        "long": OrderSide.BUY,
        "short": OrderSide.SELL,
    }
    key = (direction or "").lower()
    if key not in mapping:
        raise ValueError(f"Dirección inválida para operar: {direction}")
    return mapping[key]


def _resolve_quantity(event: dict, notional_usdt: float | None = None) -> float:
    price = _price_from_event(event)
    # Prioridad: cantidad explícita en evento -> notional USDT -> qty por defecto
    qty_raw = event.get("quantity")
    if qty_raw:
        qty = float(str(qty_raw).replace(",", "."))
        if qty <= 0:
            raise ValueError("quantity debe ser > 0")
        return qty

    notional_source = notional_usdt if notional_usdt and notional_usdt > 0 else TRADING_DEFAULT_NOTIONAL_USDT
    if notional_source > 0:
        if price is None or price <= 0:
            raise ValueError("No se puede calcular qty desde notional: precio ausente/ inválido.")
        target_notional = max(notional_source, TRADING_MIN_NOTIONAL)
        # Ajusta qty al múltiplo de step (ETHUSDT: 0.001) hacia arriba para cumplir notional mínimo
        step = 0.001
        raw_qty = target_notional / float(price)
        qty = math.ceil(raw_qty / step) * step
        if qty <= 0:
            raise ValueError("quantity calculada debe ser > 0")
        return qty

    qty = float(str(TRADING_DEFAULT_QTY).replace(",", "."))
    if qty <= 0:
        raise ValueError("quantity por defecto debe ser > 0")
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


def _has_open_position_same_direction(user_id: str, exchange: str, direction: str, symbol: str) -> bool:
    """
    Devuelve True si ya hay posición abierta en la misma dirección para el símbolo.
    Solo aplica a binance; si falla la consulta no bloquea (retorna False).
    """
    try:
        if exchange.lower() != "binance" or _account_manager is None:
            return False
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        api_key, api_secret = cred.resolve_keys(os.environ)
        base_url = "https://testnet.binancefuture.com" if cred.environment == ExchangeEnvironment.TESTNET else None
        client = UMFutures(key=api_key, secret=api_secret, base_url=base_url) if base_url else UMFutures(
            key=api_key, secret=api_secret
        )
        pos = client.get_position_risk(symbol=symbol)
        if not pos:
            return False
        pos_amt = float(pos[0].get("positionAmt") or 0)
        if direction == "long" and pos_amt > 0:
            return True
        if direction == "short" and pos_amt < 0:
            return True
        return False
    except Exception as exc:  # pragma: no cover - externo
        print(f"[WATCHER][WARN] No se pudo obtener posición para {user_id}/{exchange}: {exc}")
        return False


def _cancel_reduce_only_open(client: UMFutures, symbol: str):
    """
    Cancela órdenes reduceOnly abiertas del símbolo (TP/SL previos).
    """
    try:
        orders = client.get_all_orders(symbol=symbol, limit=200)
    except Exception as exc:  # pragma: no cover - externo
        print(f"[WATCHER][WARN] No se pudieron obtener órdenes para limpiar reduceOnly: {exc}")
        return
    for o in orders:
        try:
            if o.get("status") != "NEW":
                continue
            if not o.get("reduceOnly"):
                continue
            oid = o.get("orderId")
            client.cancel_order(symbol=symbol, orderId=oid)
        except Exception as exc:  # pragma: no cover - externo
            print(f"[WATCHER][WARN] No se pudo cancelar reduceOnly {o.get('orderId')}: {exc}")


def _interval_seconds(interval: str) -> int:
    unit = interval[-1].lower()
    value = int(interval[:-1])
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    raise ValueError(f"Intervalo no soportado: {interval}")


def _close_opposite_position(user_id: str, exchange: str, direction: str, symbol: str, price: float) -> bool:
    """
    Si hay posición abierta en dirección opuesta, intenta cerrarla con una orden reduceOnly LIMIT
    al mismo precio de la nueva señal. Devuelve True si no hay opuesta o si se pudo enviar el cierre.
    """
    try:
        if exchange.lower() != "binance" or _account_manager is None:
            return True
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        api_key, api_secret = cred.resolve_keys(os.environ)
        base_url = "https://testnet.binancefuture.com" if cred.environment == ExchangeEnvironment.TESTNET else None
        client = UMFutures(key=api_key, secret=api_secret, base_url=base_url) if base_url else UMFutures(
            key=api_key, secret=api_secret
        )
        # Si no hay posición, limpia reduceOnly pendientes del símbolo
        pos = client.get_position_risk(symbol=symbol)
        if not pos:
            _cancel_reduce_only_open(client, symbol)
            return True
        pos_amt = float(pos[0].get("positionAmt") or 0)
        if pos_amt == 0:
            _cancel_reduce_only_open(client, symbol)
            return True
        # Chequear si es opuesto
        if direction == "long" and pos_amt > 0:
            return True
        if direction == "short" and pos_amt < 0:
            return True
        qty = abs(pos_amt)
        # Ejecuta cierre reduceOnly LIMIT al precio de la nueva señal
        side = "BUY" if pos_amt < 0 else "SELL"
        try:
            client.new_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                price=f"{price:.1f}",
                quantity=f"{qty:.3f}",
                timeInForce="GTC",
                reduceOnly="true",
            )
            print(
                f"[WATCHER][INFO] Cierre reduceOnly (LIMIT) de posición opuesta qty={qty} side={side} "
                f"en {symbol} price={price:.1f}"
            )
            return True
        except Exception as exc:  # pragma: no cover - externo
            print(f"[WATCHER][ERROR] No se pudo cerrar posición opuesta ({exc})")
            return False
    except Exception as exc:  # pragma: no cover - externo
        print(f"[WATCHER][WARN] No se pudo verificar/cerrar posición opuesta: {exc}")
        return False


def _submit_trade(event: dict) -> None:
    global _last_order_direction
    executor = _resolve_executor()
    if executor is None:
        return
    targets = _resolve_targets()
    if not targets:
        print("[WATCHER][WARN] No hay cuentas habilitadas/filtradas para operar; se omite trading.")
        return
    try:
        side = _direction_to_side(event.get("direction"))
    except Exception as exc:
        print(f"[WATCHER][WARN] No se pudo determinar dirección para trading: {exc}")
        return
    direction = (event.get("direction") or "").lower()
    if direction and direction == (_last_order_direction or "").lower():
        print(f"[WATCHER][INFO] Orden {direction} ya colocada; se ignora nueva señal.")
        return

    price = _price_from_event(event)
    if price is None or price <= 0:
        print("[WATCHER][WARN] Evento sin precio de referencia, se omite trading.")
        return
    if TRADING_MIN_PRICE > 0 and price < TRADING_MIN_PRICE:
        print(f"[WATCHER][INFO] Precio {price:.2f} < mínimo configurado ({TRADING_MIN_PRICE}); no se opera.")
        return
    for user_id, exchange in targets:
        try:
            account = _account_manager.get_account(user_id) if _account_manager else None
            cred = account.get_exchange(exchange) if account else None
            notional = cred.notional_usdt if cred else None
        except Exception:
            notional = None
        try:
            quantity = _resolve_quantity({**event, "price": price}, notional_usdt=notional)
        except Exception as exc:
            print(f"[WATCHER][WARN] Cantidad inválida para trading ({exc}) usuario={user_id} exchange={exchange}")
            continue
        tp_price = event.get("tp") or event.get("take_profit")
        sl_price = event.get("sl") or event.get("stop_loss")
        try:
            tp_price = float(tp_price) if tp_price is not None else None
        except Exception:
            tp_price = None
        try:
            sl_price = float(sl_price) if sl_price is not None else None
        except Exception:
            sl_price = None
        symbol = event.get("symbol") or SYMBOL_DISPLAY.replace(".P", "")
        # Si hay posición opuesta, intenta cerrarla antes de re-entrar
        if not _close_opposite_position(user_id, exchange, direction, symbol, price):
            print(f"[WATCHER][WARN] No se pudo cerrar posición opuesta en {symbol}; se omite señal.")
            continue
        if _has_open_position_same_direction(user_id, exchange, direction, symbol):
            print(f"[WATCHER][INFO] Ya hay posición {direction} abierta en {symbol}; se omite señal.")
            continue
        order = OrderRequest(
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=TimeInForce.GTC,
            extra_params={
                "source_event": event.get("type", "unknown"),
                "event_timestamp": str(event.get("timestamp")),
                "account": user_id,
                "exchange": exchange,
                "tp": tp_price,
                "sl": sl_price,
            },
        )
        try:
            response = executor.execute(user_id, exchange, order, dry_run=TRADING_DRY_RUN)
            print(
                f"[WATCHER][TRADE] user={user_id} ex={exchange} success={response.success} status={response.status} raw={response.raw}"
            )
            try:
                # Log explícito de los TP/SL enviados (bracket) para depurar fallos.
                bracket_raw = response.raw.get("bracket") if isinstance(response.raw, dict) else None
                if bracket_raw is not None:
                    print(f"[WATCHER][BRACKET] user={user_id} ex={exchange} bracket={bracket_raw}")
            except Exception:
                # No interrumpir el loop si la respuesta no tiene el formato esperado.
                pass
            if response.success:
                _last_order_direction = direction
        except Exception as exc:
            print(f"[WATCHER][ERROR] Falló la ejecución de orden usuario={user_id} exchange={exchange} ({exc})")

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
    interval_sec = _interval_seconds(STREAM_INTERVAL)
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
            # Agenda para la siguiente vela (cierre + 1 intervalo)
            ts = evt.get("timestamp")
            try:
                if ts is None:
                    ts = datetime.now(timezone.utc)
                elif isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                elif ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                execute_at = ts + timedelta(seconds=interval_sec)
            except Exception:
                ts = datetime.now(timezone.utc)
                execute_at = ts + timedelta(seconds=interval_sec)
            _pending_signals.append({"execute_at": execute_at, "event": evt})

        # Ejecuta alertas pendientes cuyo tiempo ya venció
        now_utc = datetime.now(timezone.utc)
        due = [p for p in _pending_signals if p["execute_at"] <= now_utc]
        _pending_signals[:] = [p for p in _pending_signals if p["execute_at"] > now_utc]

        if due:
            alerts_to_send = [p["event"] for p in due]
            try:
                send_alerts(alerts_to_send)
            except Exception as exc:
                print(f"[ALERT][WARN] Falló envío de alertas ({exc})")
            if TRADING_ENABLED:
                for item in due:
                    _submit_trade(item["event"])

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
