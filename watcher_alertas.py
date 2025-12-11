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
_thresholds: list[dict] = []
THRESHOLDS_PATH = Path("backtest/backtestTR/pending_thresholds.json")
LOSS_PCT = 0.05  # 5% en contra
GAIN_PCT = 0.09  # 9% a favor


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


def _load_thresholds():
    """
    Carga umbrales pendientes desde disco (si existe).
    """
    global _thresholds
    try:
        if THRESHOLDS_PATH.exists():
            import json

            data = json.loads(THRESHOLDS_PATH.read_text())
            if isinstance(data, list):
                _thresholds = data
    except Exception:
        _thresholds = []


def _save_thresholds():
    """
    Guarda umbrales pendientes en disco para persistencia simple.
    """
    try:
        import json

        THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        THRESHOLDS_PATH.write_text(json.dumps(_thresholds, indent=2))
    except Exception:
        pass


_load_thresholds()


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


def _compute_thresholds(direction: str, entry_price: float) -> tuple[float, float]:
    """
    Calcula precios objetivo para cierre por pérdida/ganancia fija.
    Long: loss = -5%, gain = +9%
    Short: loss = +5% (en contra), gain = -9% (a favor).
    """
    if direction == "long":
        loss_price = entry_price * (1 - LOSS_PCT)
        gain_price = entry_price * (1 + GAIN_PCT)
    else:
        loss_price = entry_price * (1 + LOSS_PCT)
        gain_price = entry_price * (1 - GAIN_PCT)
    return loss_price, gain_price


def _register_threshold(user_id: str, exchange: str, symbol: str, direction: str, entry_price: float):
    """
    Registra umbrales de cierre (-5% / +9%) para una nueva operación.
    Reemplaza cualquier registro previo del mismo usuario/exchange/símbolo.
    """
    global _thresholds
    loss_price, gain_price = _compute_thresholds(direction, entry_price)
    # filtra previos
    _thresholds = [
        th
        for th in _thresholds
        if not (
            th.get("user_id") == user_id
            and th.get("exchange") == exchange
            and th.get("symbol") == symbol
        )
    ]
    _thresholds.append(
        {
            "user_id": user_id,
            "exchange": exchange,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "loss_price": loss_price,
            "gain_price": gain_price,
            "fired_loss": False,
            "fired_gain": False,
        }
    )
    _save_thresholds()


def _current_position(user_id: str, exchange: str, symbol: str) -> float:
    """
    Devuelve cantidad firmada de la posición actual (long >0, short <0).
    Solo implementado para binance; si falla devuelve 0.
    """
    try:
        if exchange.lower() != "binance" or _account_manager is None:
            return 0.0
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        api_key, api_secret = cred.resolve_keys(os.environ)
        base_url = "https://testnet.binancefuture.com" if cred.environment == ExchangeEnvironment.TESTNET else None
        client = UMFutures(key=api_key, secret=api_secret, base_url=base_url) if base_url else UMFutures(
            key=api_key, secret=api_secret
        )
        pos = client.get_position_risk(symbol=symbol)
        if not pos:
            return 0.0
        return float(pos[0].get("positionAmt") or 0.0)
    except Exception:
        return 0.0


def _close_position(user_id: str, exchange: str, symbol: str, direction: str) -> bool:
    """
    Cierra posición completa usando orden reduceOnly MARKET.
    direction: sentido de la posición actual ('long' -> vender, 'short' -> comprar)
    """
    if exchange.lower() != "binance" or _account_manager is None:
        return False
    try:
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        api_key, api_secret = cred.resolve_keys(os.environ)
        base_url = "https://testnet.binancefuture.com" if cred.environment == ExchangeEnvironment.TESTNET else None
        client = UMFutures(key=api_key, secret=api_secret, base_url=base_url) if base_url else UMFutures(
            key=api_key, secret=api_secret
        )
        pos_amt = _current_position(user_id, exchange, symbol)
        if pos_amt == 0:
            return False
        qty = abs(pos_amt)
        side = "SELL" if pos_amt > 0 else "BUY"
        client.new_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=f"{qty:.3f}",
            reduceOnly="true",
        )
        print(f"[WATCHER][INFO] Cierre reduceOnly MARKET user={user_id} ex={exchange} symbol={symbol} qty={qty} side={side}")
        return True
    except Exception as exc:
        print(f"[WATCHER][WARN] No se pudo cerrar posición user={user_id} ex={exchange}: {exc}")
        return False


def _evaluate_thresholds(current_price: float, ts) -> list[dict]:
    """
    Evalúa si el precio actual dispara algún cierre por pérdida/ganancia.
    Devuelve lista de alertas a emitir y ejecuta cierre reduceOnly MARKET cuando corresponde.
    """
    alerts = []
    updated = False
    keep_thresholds = []

    for th in _thresholds:
        user_id = th.get("user_id")
        exchange = th.get("exchange")
        symbol = th.get("symbol", SYMBOL_DISPLAY.replace(".P", ""))
        direction = th.get("direction")
        entry = float(th.get("entry_price") or 0)
        loss_price = float(th.get("loss_price") or 0)
        gain_price = float(th.get("gain_price") or 0)
        fired_loss = th.get("fired_loss", False)
        fired_gain = th.get("fired_gain", False)

        if entry <= 0:
            continue
        # Si ya no hay posición, limpiar registro
        pos_amt = _current_position(user_id, exchange, symbol)
        if pos_amt == 0:
            updated = True
            continue

        hit_loss = False
        hit_gain = False
        if direction == "long":
            hit_loss = (not fired_loss) and current_price <= loss_price
            hit_gain = (not fired_gain) and current_price >= gain_price
        else:  # short
            hit_loss = (not fired_loss) and current_price >= loss_price
            hit_gain = (not fired_gain) and current_price <= gain_price

        if hit_loss or hit_gain:
            kind = "ganancia +9%" if hit_gain else "pérdida -5%"
            # Ejecuta cierre reduceOnly MARKET del tamaño actual
            _close_position(user_id, exchange, symbol, direction)
            alerts.append(
                {
                    "type": "auto_close",
                    "timestamp": ts,
                    "message": (
                        f"{symbol} {STREAM_INTERVAL}\n"
                        f"Cierre {direction.upper()} por {kind}\n"
                        f"Entrada: {entry:.2f}\n"
                        f"Último: {current_price:.2f}"
                    ),
                    "direction": direction,
                    "user_id": user_id,
                    "exchange": exchange,
                }
            )
            updated = True
            # una vez disparado, removemos el registro (se reemplaza con la próxima operación)
            continue

        keep_thresholds.append(th)

    if updated:
        _thresholds[:] = keep_thresholds
        _save_thresholds()

    return alerts


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


def _has_opposite_position(user_id: str, exchange: str, direction: str, symbol: str) -> bool:
    """
    True si hay posición abierta en el sentido contrario.
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
        if direction == "long" and pos_amt < 0:
            return True
        if direction == "short" and pos_amt > 0:
            return True
        return False
    except Exception:
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
    Si hay posición abierta en dirección opuesta, intenta cerrarla con una orden reduceOnly MARKET.
    Devuelve True si no hay opuesta o si se pudo enviar el cierre.
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
        # Si no hay posición, no tocamos TP/SL (ya no existen o no hay riesgo)
        pos = client.get_position_risk(symbol=symbol)
        if not pos:
            return True
        pos_amt = float(pos[0].get("positionAmt") or 0)
        if pos_amt == 0:
            return True
        # Chequear si es opuesto
        if direction == "long" and pos_amt > 0:
            return True
        if direction == "short" and pos_amt < 0:
            return True
        qty = abs(pos_amt)
        # Ejecuta cierre reduceOnly MARKET
        side = "BUY" if pos_amt < 0 else "SELL"
        try:
            client.new_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=f"{qty:.3f}",
                reduceOnly="true",
            )
            print(
                f"[WATCHER][INFO] Cierre reduceOnly (MARKET) de posición opuesta qty={qty} side={side} "
                f"en {symbol}"
            )
            return True
        except Exception as exc:  # pragma: no cover - externo
            print(f"[WATCHER][ERROR] No se pudo cerrar posición opuesta ({exc})")
            return False
    except Exception as exc:  # pragma: no cover - externo
        print(f"[WATCHER][WARN] No se pudo verificar/cerrar posición opuesta: {exc}")
        return False


def _sync_brackets(user_id: str, exchange: str, symbol: str, direction: str, tp: float | None, sl: float | None):
    """
    Recalibra TP/SL reduceOnly según la posición actual:
    - Si posición = 0: cancela reduceOnly abiertos (limpieza).
    - Si posición en el mismo sentido que direction: cancela reduceOnly y los recrea con el tamaño actual.
    - Si posición contraria: no hace nada (mantiene protección previa).
    """
    if exchange.lower() != "binance" or _account_manager is None:
        return
    if tp is None and sl is None:
        return
    try:
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        api_key, api_secret = cred.resolve_keys(os.environ)
        base_url = "https://testnet.binancefuture.com" if cred.environment == ExchangeEnvironment.TESTNET else None
        client = UMFutures(key=api_key, secret=api_secret, base_url=base_url) if base_url else UMFutures(
            key=api_key, secret=api_secret
        )
        # Estado actual de la posición
        pos = client.get_position_risk(symbol=symbol)
        pos_amt = float(pos[0].get("positionAmt") or 0) if pos else 0.0
        # Dirección de la posición actual
        if pos_amt == 0:
            _cancel_reduce_only_open(client, symbol)
            return
        if direction == "long" and pos_amt < 0:
            # Opuesta: mantenemos TP/SL existentes
            return
        if direction == "short" and pos_amt > 0:
            return
        qty = abs(pos_amt)
        # Limpia reduceOnly previos y recrea
        _cancel_reduce_only_open(client, symbol)
        side = "BUY" if pos_amt > 0 else "SELL"
        try:
            _ = client.new_order(
                symbol=symbol,
                side="SELL" if side == "BUY" else "BUY",
                type="TAKE_PROFIT",
                price=f"{tp:.1f}" if tp else None,
                stopPrice=f"{tp:.1f}" if tp else None,
                quantity=f"{qty:.3f}",
                reduceOnly="true",
                timeInForce="GTC",
            ) if tp else None
        except Exception as exc:
            print(f"[WATCHER][WARN] Error recreando TP reduceOnly: {exc}")
        try:
            _ = client.new_order(
                symbol=symbol,
                side="SELL" if side == "BUY" else "BUY",
                type="STOP",
                price=f"{sl:.1f}" if sl else None,
                stopPrice=f"{sl:.1f}" if sl else None,
                quantity=f"{qty:.3f}",
                reduceOnly="true",
                timeInForce="GTC",
            ) if sl else None
        except Exception as exc:
            print(f"[WATCHER][WARN] Error recreando SL reduceOnly: {exc}")
    except Exception as exc:  # pragma: no cover - externo
        print(f"[WATCHER][WARN] No se pudo sincronizar TP/SL: {exc}")


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
            notional = None
            if cred:
                if exchange.lower() == "dydx":
                    notional = cred.notional_usdc if cred.notional_usdc is not None else cred.notional_usdt
                else:
                    notional = cred.notional_usdt
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
        if cred and cred.extra:
            symbol = cred.extra.get("symbol", symbol)
        had_opposite = _has_opposite_position(user_id, exchange, direction, symbol)
        # Si hay posición opuesta, envía cierre reduceOnly y entrada simultánea en el mismo precio.
        # Se mantienen TP/SL previos hasta que el cierre se ejecute.
        if had_opposite:
            if not _close_opposite_position(user_id, exchange, direction, symbol, price):
                print(f"[WATCHER][WARN] No se pudo cerrar posición opuesta en {symbol}; se omite señal.")
                continue
        if _has_open_position_same_direction(user_id, exchange, direction, symbol):
            print(f"[WATCHER][INFO] Ya hay posición {direction} abierta en {symbol}; se omite señal.")
            continue

        extra = {
            "source_event": event.get("type", "unknown"),
            "event_timestamp": str(event.get("timestamp")),
            "account": user_id,
            "exchange": exchange,
            "tp": tp_price,
            "sl": sl_price,
            "margin_mode": getattr(cred, "margin_mode", None) if cred else None,
            # Cuando había posición opuesta, evitamos cancelar TP/SL previos hasta que cierre.
            "skip_bracket": had_opposite,
        }
        order = OrderRequest(
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=TimeInForce.GTC,
            extra_params=extra,
        )

        try:
            response = executor.execute(user_id, exchange, order, dry_run=TRADING_DRY_RUN)
            print(
                f"[WATCHER][TRADE] user={user_id} ex={exchange} success={response.success} status={response.status} raw={response.raw}"
            )
            try:
                bracket_raw = response.raw.get("bracket") if isinstance(response.raw, dict) else None
                if bracket_raw is not None:
                    print(f"[WATCHER][BRACKET] user={user_id} ex={exchange} bracket={bracket_raw}")
            except Exception:
                pass
            if response.success:
                _last_order_direction = direction
                # Registra umbrales fijos (-5% / +9%) para alertas de cierre
                entry_used = float(response.avg_price or price)
                _register_threshold(user_id, exchange, symbol, direction, entry_used)
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
            # Usa el precio del primer evento pendiente como referencia para evaluar umbrales
            try:
                current_price = _price_from_event(alerts_to_send[0]) if alerts_to_send else None
                if current_price:
                    ts_eval = alerts_to_send[0].get("timestamp", datetime.now(timezone.utc))
                    alerts_to_send.extend(_evaluate_thresholds(current_price, ts_eval))
            except Exception:
                pass
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
