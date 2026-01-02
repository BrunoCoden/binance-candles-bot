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
from trading.accounts.models import ExchangeEnvironment, ExchangeCredential
from trading.orders.executor import OrderExecutor
from trading.orders.models import OrderRequest, OrderSide, OrderType, TimeInForce

TRADING_ENABLED = os.getenv("WATCHER_ENABLE_TRADING", "false").lower() == "true"
TRADING_ACCOUNTS_FILE = os.getenv("WATCHER_ACCOUNTS_FILE", "trading/accounts/sample_accounts.yaml")
TRADING_USER_ID = os.getenv("WATCHER_TRADING_USER", "").strip()
TRADING_EXCHANGE = os.getenv("WATCHER_TRADING_EXCHANGE", "").strip()
TRADING_DRY_RUN = os.getenv("WATCHER_TRADING_DRY_RUN", "true").lower() != "false"
TRADING_MIN_PRICE = float(os.getenv("WATCHER_TRADING_MIN_PRICE", "0"))

_executor: OrderExecutor | None = None
_account_manager: AccountManager | None = None
_accounts_mtime: float | None = None
_last_order_direction: dict[tuple[str, str], str] = {}
_thresholds: list[dict] = []
THRESHOLDS_PATH = Path("backtest/backtestTR/pending_thresholds.json")
LOSS_PCT = 0.05  # 5% en contra
GAIN_PCT = 0.09  # 9% a favor
ACCOUNTS_AUTO_RELOAD = os.getenv("WATCHER_ACCOUNTS_AUTO_RELOAD", "false").lower() == "true"
DISABLED_ACCOUNTS_AUTO_CLOSE = os.getenv("WATCHER_DISABLED_AUTO_CLOSE", "true").lower() == "true"
DISABLED_ACCOUNTS_CLOSE_POLL_SECONDS = float(os.getenv("WATCHER_DISABLED_CLOSE_POLL_SECONDS", "30"))

_last_disabled_close_attempt: dict[tuple[str, str, str], float] = {}


def _load_manager() -> AccountManager | None:
    global _account_manager, _accounts_mtime, _executor
    if not TRADING_ENABLED:
        return None
    try:
        path = Path(TRADING_ACCOUNTS_FILE)
        try:
            current_mtime = path.stat().st_mtime
        except FileNotFoundError:
            current_mtime = None

        if not ACCOUNTS_AUTO_RELOAD and _account_manager is not None:
            return _account_manager

        if _account_manager is not None and _accounts_mtime is not None and current_mtime == _accounts_mtime:
            return _account_manager

        _account_manager = AccountManager.from_file(path)
        _accounts_mtime = current_mtime
        _executor = None
        print(f"[WATCHER][INFO] Cuentas recargadas desde {path} (mtime={_accounts_mtime})")
        return _account_manager
    except Exception as exc:
        print(f"[WATCHER][WARN] No se pudo inicializar AccountManager ({exc}); modo trading deshabilitado.")
        return None


def _bybit_position_amount(cred: ExchangeCredential, symbol: str) -> float:
    """
    Devuelve cantidad firmada (long >0, short <0) para Bybit linear.
    """
    try:
        from pybit.unified_trading import HTTP  # type: ignore

        api_key, api_secret = cred.resolve_keys(os.environ)
        is_testnet = cred.environment != ExchangeEnvironment.LIVE
        domain_env = os.getenv("BYBIT_DOMAIN_TESTNET" if is_testnet else "BYBIT_DOMAIN")
        client = (
            HTTP(api_key=api_key, api_secret=api_secret, testnet=False, domain=domain_env)
            if domain_env
            else HTTP(api_key=api_key, api_secret=api_secret, testnet=is_testnet)
        )
        raw = client.get_positions(category="linear", symbol=symbol)
        items = raw.get("result", {}).get("list") or []
        if not items:
            return 0.0
        # Unified: size + side
        pos = items[0]
        size = float(pos.get("size") or 0.0)
        side = str(pos.get("side") or "").lower()
        if size == 0:
            return 0.0
        return size if side == "buy" else -size
    except Exception:
        return 0.0


def _close_disabled_accounts_positions() -> None:
    """
    Si un usuario está enabled=false, intenta cerrar posiciones abiertas en todos sus exchanges.
    Solo actúa si WATCHER_DISABLED_AUTO_CLOSE=true.
    """
    if not TRADING_ENABLED or not DISABLED_ACCOUNTS_AUTO_CLOSE:
        return
    manager = _load_manager()
    if manager is None:
        return
    executor = _resolve_executor()
    if executor is None:
        return

    now = time.time()
    for account in manager.list_accounts():
        if account.enabled:
            continue
        for exchange, cred in (account.exchanges or {}).items():
            symbol = (cred.extra or {}).get("symbol") or SYMBOL_DISPLAY.replace(".P", "")
            try:
                pos_amt = _current_position(account.user_id, exchange, symbol)
                # fallback específico Bybit si el helper legacy no lo soporta
                if pos_amt == 0 and exchange.lower() == "bybit":
                    pos_amt = _bybit_position_amount(cred, symbol)
            except Exception:
                pos_amt = 0.0
            if not pos_amt:
                continue

            key = (account.user_id, exchange.lower(), str(symbol))
            last = _last_disabled_close_attempt.get(key, 0.0)
            if now - last < max(DISABLED_ACCOUNTS_CLOSE_POLL_SECONDS, 10.0):
                continue
            _last_disabled_close_attempt[key] = now

            qty = abs(float(pos_amt))
            side = OrderSide.SELL if pos_amt > 0 else OrderSide.BUY
            order = OrderRequest(
                symbol=symbol,
                side=side,
                type=OrderType.MARKET,
                quantity=qty,
                price=None,
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
                extra_params={
                    "source_event": "disabled_auto_close",
                    "account": account.user_id,
                    "exchange": exchange,
                },
            )
            try:
                resp = executor.execute(account.user_id, exchange, order, dry_run=TRADING_DRY_RUN)
                print(
                    f"[WATCHER][AUTO_CLOSE_DISABLED] user={account.user_id} ex={exchange} symbol={symbol} "
                    f"qty={qty} side={side.value} success={resp.success} status={resp.status}"
                )
            except Exception as exc:
                print(f"[WATCHER][WARN] Auto-cierre por disabled falló user={account.user_id} ex={exchange}: {exc}")
    try:
        path = Path(TRADING_ACCOUNTS_FILE)
        try:
            current_mtime = path.stat().st_mtime
        except FileNotFoundError:
            current_mtime = None
        if not ACCOUNTS_AUTO_RELOAD and _account_manager is not None:
            return _account_manager
        if _account_manager is not None and _accounts_mtime is not None and current_mtime == _accounts_mtime:
            return _account_manager
        _account_manager = AccountManager.from_file(path)
        _accounts_mtime = current_mtime
        _executor = None
        print(f"[WATCHER][INFO] Cuentas recargadas desde {path} (mtime={_accounts_mtime})")
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
    # Prioridad: cantidad explícita en evento -> notional USDT (desde DashCRUD/YAML por usuario/exchange)
    qty_raw = event.get("quantity")
    if qty_raw:
        qty = float(str(qty_raw).replace(",", "."))
        if qty <= 0:
            raise ValueError("quantity debe ser > 0")
        return qty

    if notional_usdt is None or notional_usdt <= 0:
        raise ValueError("Sin notional_usdt (configurarlo por usuario/exchange en DashCRUD).")
    if price is None or price <= 0:
        raise ValueError("No se puede calcular qty desde notional: precio ausente/ inválido.")
    qty = float(notional_usdt) / float(price)
    if qty <= 0:
        raise ValueError("quantity calculada debe ser > 0")
    return qty


def _price_from_event(event: dict) -> float | None:
    # Prioridad: entry/price explícitos, luego banda de referencia y, por último, close
    for key in ("entry_price", "price", "reference_band"):
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
    Implementado para binance/dydx/bybit; si falla devuelve 0.
    """
    try:
        if _account_manager is None:
            return 0.0
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        if exchange.lower() == "binance":
            api_key, api_secret = cred.resolve_keys(os.environ)
            base_url = "https://testnet.binancefuture.com" if cred.environment == ExchangeEnvironment.TESTNET else None
            client = UMFutures(key=api_key, secret=api_secret, base_url=base_url) if base_url else UMFutures(
                key=api_key, secret=api_secret
            )
            pos = client.get_position_risk(symbol=symbol)
            if not pos:
                return 0.0
            return float(pos[0].get("positionAmt") or 0.0)
        elif exchange.lower() == "dydx":
            # Usa wallet address + private key (formato v4 nativo)
            from trading.exchanges.dydx import get_dydx_position
            
            api_key, _ = cred.resolve_keys(os.environ)  # api_key es la wallet address
            subaccount_number = cred.extra.get("subaccount", 0) if cred.extra else 0
            market_symbol = cred.extra.get("symbol", symbol) if cred.extra else symbol
            
            return get_dydx_position(api_key, market_symbol, subaccount_number)
        elif exchange.lower() == "bybit":
            return _bybit_position_amount(cred, symbol)
        return 0.0
    except Exception:
        return 0.0


def _close_position(user_id: str, exchange: str, symbol: str, direction: str) -> bool:
    """
    Cierra posición completa usando orden reduceOnly MARKET.
    direction: sentido de la posición actual ('long' -> vender, 'short' -> comprar)
    """
    if _account_manager is None:
        return False
    try:
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        pos_amt = _current_position(user_id, exchange, symbol)
        if pos_amt == 0:
            return False
        qty = abs(pos_amt)
        if exchange.lower() == "binance":
            api_key, api_secret = cred.resolve_keys(os.environ)
            base_url = "https://testnet.binancefuture.com" if cred.environment == ExchangeEnvironment.TESTNET else None
            client = UMFutures(key=api_key, secret=api_secret, base_url=base_url) if base_url else UMFutures(
                key=api_key, secret=api_secret
            )
            side = "SELL" if pos_amt > 0 else "BUY"
            client.new_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=f"{qty:.3f}",
                reduceOnly="true",
            )
        elif exchange.lower() == "dydx":
            # Usa OrderExecutor con wallet address + private key (formato v4 nativo)
            from trading.exchanges.dydx import close_dydx_position_via_order_executor
            
            market_symbol = cred.extra.get("symbol", symbol) if cred.extra else symbol
            success = close_dydx_position_via_order_executor(account, cred, market_symbol, pos_amt)
            if not success:
                return False
            side = "BUY" if pos_amt < 0 else "SELL"
        else:
            return False
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
        if _account_manager is None:
            return False
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        pos_amt = _current_position(user_id, exchange, symbol)
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
        if _account_manager is None:
            return False
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        pos_amt = _current_position(user_id, exchange, symbol)
        if direction == "long" and pos_amt < 0:
            return True
        if direction == "short" and pos_amt > 0:
            return True
        return False
    except Exception:
        return False


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
        if _account_manager is None:
            return True
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        pos_amt = _current_position(user_id, exchange, symbol)
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
        if exchange.lower() == "binance":
            api_key, api_secret = cred.resolve_keys(os.environ)
            base_url = "https://testnet.binancefuture.com" if cred.environment == ExchangeEnvironment.TESTNET else None
            client = UMFutures(key=api_key, secret=api_secret, base_url=base_url) if base_url else UMFutures(
                key=api_key, secret=api_secret
            )
            client.new_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=f"{qty:.3f}",
                reduceOnly="true",
            )
        elif exchange.lower() == "dydx":
            # Usa OrderExecutor con wallet address + private key (formato v4 nativo)
            from trading.exchanges.dydx import close_dydx_position_via_order_executor
            
            market_symbol = cred.extra.get("symbol", symbol) if cred.extra else symbol
            # pos_amt ya tiene el signo correcto (positivo long, negativo short)
            success = close_dydx_position_via_order_executor(account, cred, market_symbol, pos_amt)
            if not success:
                return False
        print(
            f"[WATCHER][INFO] Cierre reduceOnly (MARKET) de posición opuesta qty={qty} side={side} en {symbol} ex={exchange}"
        )
        return True
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
        symbol = event.get("symbol") or SYMBOL_DISPLAY.replace(".P", "")
        if cred and cred.extra:
            symbol = cred.extra.get("symbol", symbol)

        # Nota: no hay overrides por env para el monto. El sizing sale del YAML (DashCRUD) por usuario/exchange.
        had_opposite = _has_opposite_position(user_id, exchange, direction, symbol)
        # Si hay posición opuesta, envía cierre reduceOnly y entrada simultánea en el mismo precio.
        # Se mantienen TP/SL previos hasta que el cierre se ejecute.
        if had_opposite:
            if not _close_opposite_position(user_id, exchange, direction, symbol, price):
                print(f"[WATCHER][WARN] No se pudo cerrar posición opuesta en {symbol}; se omite señal.")
                continue
        key_dir = (user_id, exchange)
        last_dir = _last_order_direction.get(key_dir)
        if direction and direction == (last_dir or "").lower():
            print(f"[WATCHER][INFO] Orden {direction} ya colocada en {exchange}; se ignora señal.")
            continue
        if _has_open_position_same_direction(user_id, exchange, direction, symbol):
            print(f"[WATCHER][INFO] Ya hay posición {direction} abierta en {symbol}; se omite señal.")
            continue

        # Chequeo de tope de posición si se configuró
        try:
            if cred and cred.max_position_usdc and price:
                pos_amt = _current_position(user_id, exchange, symbol)
                current_notional = abs(pos_amt) * price
                next_notional = quantity * price
                if current_notional + next_notional > cred.max_position_usdc:
                    print(
                        f"[WATCHER][INFO] Tope de posición excedido ({current_notional + next_notional:.2f} > {cred.max_position_usdc}); se omite señal."
                    )
                    continue
        except Exception:
            pass

        extra = {
            "source_event": event.get("type", "unknown"),
            "event_timestamp": str(event.get("timestamp")),
            "account": user_id,
            "exchange": exchange,
            "margin_mode": getattr(cred, "margin_mode", None) if cred else None,
        }
        order = OrderRequest(
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            quantity=quantity,
            price=None,
            time_in_force=TimeInForce.GTC,
            extra_params=extra,
        )

        try:
            response = executor.execute(user_id, exchange, order, dry_run=TRADING_DRY_RUN)
            err = getattr(response, "error", None)
            err_text = f" error={err}" if err else ""
            print(
                f"[WATCHER][TRADE] user={user_id} ex={exchange} success={response.success} status={response.status}{err_text} raw={response.raw}"
            )
            if response.success:
                _last_order_direction[key_dir] = direction
                # Registra umbrales fijos (-5% / +9%) para alertas de cierre
                entry_used = float(response.avg_price or price)
                _register_threshold(user_id, exchange, symbol, direction, entry_used)
        except RuntimeError as exc:
            # Faltan credenciales u otro error de configuración: loguea y sigue con el siguiente exchange/usuario.
            print(f"[WATCHER][WARN] Credenciales/config faltantes para {user_id}/{exchange}: {exc}")
            continue
        except Exception as exc:
            print(f"[WATCHER][ERROR] Falló la ejecución de orden usuario={user_id} exchange={exchange} ({exc})")

POLL_SECONDS = float(os.getenv("ALERT_POLL_SECONDS", "5"))
MAX_SEEN = int(os.getenv("ALERT_MAX_SEEN", "500"))
SEND_STARTUP_TEST = os.getenv("WATCHER_STARTUP_TEST_ALERT", "true").lower() == "true"
THRESHOLD_POLL_SECONDS = float(os.getenv("THRESHOLD_POLL_SECONDS", "1"))


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
    last_threshold_check = 0.0
    last_disabled_check = 0.0
    # Al iniciar, si hay cuentas disabled, intenta cerrar posiciones abiertas.
    try:
        _close_disabled_accounts_positions()
    except Exception:
        pass
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
            alerts_to_send = [evt]
            # Evalúa umbrales de cierre con el precio de la señal actual
            try:
                current_price = _price_from_event(evt)
                if current_price:
                    ts_eval = evt.get("timestamp", datetime.now(timezone.utc))
                    alerts_to_send.extend(_evaluate_thresholds(current_price, ts_eval))
            except Exception:
                pass
            print(f"[ALERTA] {format_alert_message(evt)}")
            try:
                send_alerts(alerts_to_send)
            except Exception as exc:
                print(f"[ALERT][WARN] Falló envío de alertas ({exc})")
            if TRADING_ENABLED:
                _submit_trade(evt)

        # Chequeo periódico de umbrales aunque no haya nuevas alertas
        now_ts = time.time()
        # Chequeo periódico de cuentas disabled (auto-cierre)
        if now_ts - last_disabled_check >= DISABLED_ACCOUNTS_CLOSE_POLL_SECONDS:
            last_disabled_check = now_ts
            try:
                _close_disabled_accounts_positions()
            except Exception:
                pass
        if now_ts - last_threshold_check >= THRESHOLD_POLL_SECONDS:
            last_threshold_check = now_ts
            try:
                # intenta obtener precio desde la última vela en stream (si generate_alerts lo dejó en cache)
                # fallback: usa el precio del último evento pendiente o el último precio conocido en ALERTS_TABLE_CSV_PATH si fuese necesario
                current_price = None
                try:
                    from velas import LAST_KLINES_CACHE  # type: ignore
                    df = LAST_KLINES_CACHE.get("stream") if isinstance(LAST_KLINES_CACHE, dict) else None
                    if df is not None and not df.empty:
                        current_price = float(df["Close"].iloc[-1])
                except Exception:
                    pass
                if current_price:
                    ts_eval = datetime.now(timezone.utc)
                    try:
                        extra_alerts = _evaluate_thresholds(current_price, ts_eval)
                        if extra_alerts:
                            send_alerts(extra_alerts)
                    except Exception as exc:
                        print(f"[ALERT][WARN] Falló evaluación periódica de umbrales ({exc})")
            except Exception:
                pass

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
