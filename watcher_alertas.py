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
LOSS_PCT = float(os.getenv("WATCHER_CONTRA_THRESHOLD_PCT", "0.02"))  # 2% en contra
GAIN_PCT = 0.0  # sin TP en esta lógica
THRESHOLDS_CLEAR_ON_STARTUP = os.getenv("WATCHER_THRESHOLDS_CLEAR_ON_STARTUP", "false").lower() == "true"
THRESHOLDS_REBUILD_ON_STARTUP = os.getenv("WATCHER_THRESHOLDS_REBUILD_ON_STARTUP", "false").lower() == "true"
ACCOUNTS_AUTO_RELOAD = os.getenv("WATCHER_ACCOUNTS_AUTO_RELOAD", "false").lower() == "true"
DISABLED_ACCOUNTS_AUTO_CLOSE = os.getenv("WATCHER_DISABLED_AUTO_CLOSE", "true").lower() == "true"
DISABLED_ACCOUNTS_CLOSE_POLL_SECONDS = float(os.getenv("WATCHER_DISABLED_CLOSE_POLL_SECONDS", "30"))
CLOSE_OPPOSITE_TIMEOUT_SECONDS = float(os.getenv("WATCHER_CLOSE_OPPOSITE_TIMEOUT_SECONDS", "10"))
CLOSE_OPPOSITE_POLL_SECONDS = float(os.getenv("WATCHER_CLOSE_OPPOSITE_POLL_SECONDS", "0.5"))

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


def _bybit_position_amount(cred: ExchangeCredential, symbol: str) -> float | None:
    """
    Devuelve cantidad firmada (long >0, short <0) para Bybit linear.
    Si hay error en la API devuelve None (posición desconocida).
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
        return None


def _bybit_mark_price(cred: ExchangeCredential, symbol: str) -> float | None:
    """
    Obtiene mark price desde Bybit (fallback: last price).
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
        raw = client.get_tickers(category="linear", symbol=symbol)
        items = raw.get("result", {}).get("list") or []
        if not items:
            return None
        data = items[0] or {}
        for key in ("markPrice", "indexPrice", "lastPrice"):
            val = data.get(key)
            if val:
                return float(val)
    except Exception:
        return None
    return None


def _binance_mark_price(cred: ExchangeCredential, symbol: str) -> float | None:
    """
    Obtiene mark price desde Binance (preferido para triggers en tiempo real).
    """
    try:
        api_key, api_secret = cred.resolve_keys(os.environ)
    except Exception:
        api_key = None
        api_secret = None
    try:
        base_url = "https://testnet.binancefuture.com" if cred.environment == ExchangeEnvironment.TESTNET else None
        if api_key and api_secret:
            client = UMFutures(key=api_key, secret=api_secret, base_url=base_url) if base_url else UMFutures(
                key=api_key, secret=api_secret
            )
        else:
            client = UMFutures(base_url=base_url) if base_url else UMFutures()
        data = client.mark_price(symbol=symbol)
        if isinstance(data, dict):
            price = data.get("markPrice") or data.get("indexPrice")
            if price:
                return float(price)
    except Exception:
        return None
    return None


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
            except Exception:
                pos_amt = None
            if pos_amt is None:
                continue
            # fallback específico Bybit si el helper legacy no lo soporta
            if pos_amt == 0 and exchange.lower() == "bybit":
                pos_amt = _bybit_position_amount(cred, symbol)
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

def _clear_thresholds_file() -> None:
    global _thresholds
    _thresholds = []
    try:
        THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _save_thresholds()
        print(f"[WATCHER][THRESHOLDS] Limpiado archivo de umbrales ([]) en: {THRESHOLDS_PATH}")
    except Exception as exc:
        print(f"[WATCHER][WARN] No se pudo limpiar archivo de umbrales {THRESHOLDS_PATH}: {exc}")


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


def _opposite_direction(direction: str) -> str:
    if direction == "long":
        return "short"
    if direction == "short":
        return "long"
    raise ValueError(f"Dirección inválida para invertir: {direction}")


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
    Calcula precio de cierre por pérdida fija (2%).
    Long: loss = -2%
    Short: loss = +2% (en contra)
    """
    if direction == "long":
        loss_price = entry_price * (1 - LOSS_PCT)
    else:
        loss_price = entry_price * (1 + LOSS_PCT)
    gain_price = None
    return loss_price, gain_price


def _register_threshold(
    user_id: str, exchange: str, symbol: str, direction: str, entry_price: float, signal_direction: str | None
):
    """
    Registra umbral de cierre (-2%) para una nueva operación.
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
            "signal_direction": signal_direction,
            "entry_price": entry_price,
            "loss_price": loss_price,
            "gain_price": gain_price,
            "fired_loss": False,
            "fired_gain": False,
        }
    )
    print(
        f"[WATCHER][THRESHOLDS][REGISTER] user={user_id} ex={exchange} symbol={symbol} dir={direction} "
        f"entry={entry_price:.6f} loss={loss_price:.6f} gain={gain_price}"
    )
    _save_thresholds()


def _update_threshold_from_signal(
    user_id: str,
    exchange: str,
    symbol: str,
    position_direction: str,
    signal_direction: str,
    entry_price: float,
) -> None:
    global _thresholds
    loss_price, gain_price = _compute_thresholds(position_direction, entry_price)
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
            "direction": position_direction,
            "signal_direction": signal_direction,
            "entry_price": entry_price,
            "loss_price": loss_price,
            "gain_price": gain_price,
            "fired_loss": False,
            "fired_gain": False,
        }
    )
    print(
        f"[WATCHER][THRESHOLDS][UPDATE] user={user_id} ex={exchange} symbol={symbol} dir={position_direction} "
        f"entry={entry_price:.6f} loss={loss_price:.6f}"
    )
    _save_thresholds()


def _execute_trade_for_target(
    user_id: str,
    exchange: str,
    direction: str,
    symbol: str,
    price: float,
    source_event: str,
    signal_direction: str | None = None,
) -> tuple[bool, float | None]:
    executor = _resolve_executor()
    if executor is None:
        return False, None
    account = _account_manager.get_account(user_id) if _account_manager else None
    cred = account.get_exchange(exchange) if account else None
    notional = None
    if cred:
        notional = cred.notional_usdt
        if cred.extra:
            symbol = cred.extra.get("symbol", symbol)
    try:
        quantity = _resolve_quantity({"price": price}, notional_usdt=notional)
    except Exception as exc:
        print(
            f"[WATCHER][WARN] Cantidad inválida para trading ({exc}) usuario={user_id} exchange={exchange}"
        )
        return False, None
    if exchange.lower() == "binance":
        step = 0.001
        quantity = math.ceil(quantity / step) * step
    side = _direction_to_side(direction)
    extra = {
        "source_event": source_event,
        "account": user_id,
        "exchange": exchange,
        "signal_direction": signal_direction,
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
    response = executor.execute(user_id, exchange, order, dry_run=TRADING_DRY_RUN)
    err = getattr(response, "error", None)
    err_text = f" error={err}" if err else ""
    print(
        f"[WATCHER][TRADE] user={user_id} ex={exchange} success={response.success} status={response.status}{err_text} raw={response.raw}"
    )
    if response.success:
        _last_order_direction[(user_id, exchange)] = direction
        return True, float(response.avg_price or price)
    return False, None


def _binance_position_details(cred: ExchangeCredential, symbol: str) -> tuple[float, float | None]:
    """
    Devuelve (position_amt_signed, entry_price) para Binance Futures.
    """
    try:
        api_key, api_secret = cred.resolve_keys(os.environ)
        base_url = "https://testnet.binancefuture.com" if cred.environment == ExchangeEnvironment.TESTNET else None
        client = UMFutures(key=api_key, secret=api_secret, base_url=base_url) if base_url else UMFutures(
            key=api_key, secret=api_secret
        )
        pos = client.get_position_risk(symbol=symbol)
        if not pos:
            return 0.0, None
        row = pos[0] or {}
        amt = float(row.get("positionAmt") or 0.0)
        entry = row.get("entryPrice")
        try:
            entry_price = float(entry) if entry is not None else None
        except Exception:
            entry_price = None
        if entry_price is not None and entry_price <= 0:
            entry_price = None
        return amt, entry_price
    except Exception:
        return 0.0, None


def _bybit_position_details(cred: ExchangeCredential, symbol: str) -> tuple[float, float | None]:
    """
    Devuelve (position_amt_signed, entry_price) para Bybit linear.
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
            return 0.0, None
        pos = items[0] or {}
        size = float(pos.get("size") or 0.0)
        side = str(pos.get("side") or "").lower()
        if size == 0:
            return 0.0, None
        signed_amt = size if side == "buy" else -size
        entry_price = None
        for k in ("avgPrice", "entryPrice", "avgEntryPrice"):
            v = pos.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
                if fv > 0:
                    entry_price = fv
                    break
            except Exception:
                continue
        return signed_amt, entry_price
    except Exception:
        return 0.0, None


def _rebuild_thresholds_from_open_positions() -> None:
    """
    Recalcula umbrales (-2%) en base a las posiciones abiertas actuales.
    - Requiere acceso a exchanges (no dry-run).
    - Si no se puede obtener entry_price, omite ese par y deja log.
    """
    if not TRADING_ENABLED:
        print("[WATCHER][THRESHOLDS][REBUILD] Trading deshabilitado; no se reconstruyen umbrales.")
        return
    manager = _load_manager()
    if manager is None:
        print("[WATCHER][THRESHOLDS][REBUILD] No hay AccountManager; no se reconstruyen umbrales.")
        return

    global _thresholds
    rebuilt = 0
    skipped = 0
    missing_entry = 0
    kept_existing = 0
    scanned = 0

    existing = {}
    for th in _thresholds:
        key = (th.get("user_id"), th.get("exchange"), th.get("symbol"))
        if all(key):
            existing[key] = th

    new_thresholds: list[dict] = []

    for account in manager.list_accounts():
        if not account.enabled:
            continue
        for exchange, cred in (account.exchanges or {}).items():
            scanned += 1
            symbol = (cred.extra or {}).get("symbol") or SYMBOL_DISPLAY.replace(".P", "")
            pos_amt = 0.0
            entry_price = None
            ex_l = exchange.lower()
            try:
                if ex_l == "binance":
                    pos_amt, entry_price = _binance_position_details(cred, symbol)
                elif ex_l == "bybit":
                    pos_amt, entry_price = _bybit_position_details(cred, symbol)
                else:
                    continue
            except Exception as exc:
                print(
                    f"[WATCHER][THRESHOLDS][REBUILD][ERR] user={account.user_id} ex={exchange} symbol={symbol} "
                    f"err={exc}"
                )
                skipped += 1
                continue

            if not pos_amt:
                print(
                    f"[WATCHER][THRESHOLDS][REBUILD][NO_POS] user={account.user_id} ex={exchange} symbol={symbol}"
                )
                continue
            direction = "long" if pos_amt > 0 else "short"
            key = (account.user_id, exchange, symbol)
            if entry_price is None or entry_price <= 0:
                missing_entry += 1
                prev = existing.get(key)
                prev_entry = float(prev.get("entry_price") or 0) if prev else 0.0
                if prev_entry > 0:
                    loss_price, gain_price = _compute_thresholds(direction, prev_entry)
                    new_thresholds.append(
                        {
                            "user_id": account.user_id,
                            "exchange": exchange,
                            "symbol": symbol,
                            "direction": direction,
                            "signal_direction": prev.get("signal_direction") if prev else direction,
                            "entry_price": prev_entry,
                            "loss_price": loss_price,
                            "gain_price": gain_price,
                            "fired_loss": False,
                            "fired_gain": False,
                        }
                    )
                    kept_existing += 1
                    print(
                        f"[WATCHER][THRESHOLDS][REBUILD][KEEP] user={account.user_id} ex={exchange} "
                        f"symbol={symbol} entry={prev_entry:.6f} reason=no_entry_price"
                    )
                else:
                    print(
                        f"[WATCHER][THRESHOLDS][REBUILD][SKIP] user={account.user_id} ex={exchange} "
                        f"symbol={symbol} pos_amt={pos_amt} reason=no_entry_price"
                    )
                    skipped += 1
                continue

            entry_val = float(entry_price)
            loss_price, gain_price = _compute_thresholds(direction, entry_val)
            new_thresholds.append(
                {
                    "user_id": account.user_id,
                    "exchange": exchange,
                    "symbol": symbol,
                    "direction": direction,
                    "signal_direction": direction,
                    "entry_price": entry_val,
                    "loss_price": loss_price,
                    "gain_price": gain_price,
                    "fired_loss": False,
                    "fired_gain": False,
                }
            )
            rebuilt += 1

    _thresholds = new_thresholds
    _save_thresholds()
    removed = max(len(existing) - len(new_thresholds), 0)
    print(
        f"[WATCHER][THRESHOLDS][REBUILD] done scanned={scanned} rebuilt={rebuilt} kept={kept_existing} "
        f"missing_entry={missing_entry} skipped={skipped} removed={removed}"
    )


def _current_position(user_id: str, exchange: str, symbol: str) -> float | None:
    """
    Devuelve cantidad firmada de la posición actual (long >0, short <0).
    Implementado para binance/bybit; si falla devuelve None.
    """
    try:
        # En dry-run no consultamos exchanges (evita requests reales en simulaciones).
        if TRADING_DRY_RUN:
            return 0.0
        if _account_manager is None:
            return None
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
        elif exchange.lower() == "bybit":
            return _bybit_position_amount(cred, symbol)
        return 0.0
    except Exception:
        return None


def _close_position(user_id: str, exchange: str, symbol: str, direction: str) -> bool:
    """
    Cierra posición completa usando orden reduceOnly MARKET.
    direction: sentido de la posición actual ('long' -> vender, 'short' -> comprar)
    """
    if _account_manager is None:
        return False
    if TRADING_DRY_RUN:
        print(
            f"[WATCHER][INFO] Dry-run activo: se omite cierre real user={user_id} ex={exchange} symbol={symbol}"
        )
        return True
    try:
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        pos_amt = _current_position(user_id, exchange, symbol)
        if pos_amt is None:
            return False
        if pos_amt is None:
            return False
        if pos_amt is None:
            return False
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
        elif exchange.lower() == "bybit":
            from decimal import Decimal, ROUND_DOWN, ROUND_UP
            import re
            from pybit.unified_trading import HTTP  # type: ignore

            api_key, api_secret = cred.resolve_keys(os.environ)
            is_testnet = cred.environment != ExchangeEnvironment.LIVE
            domain_env = os.getenv("BYBIT_DOMAIN_TESTNET" if is_testnet else "BYBIT_DOMAIN")
            client = (
                HTTP(api_key=api_key, api_secret=api_secret, testnet=False, domain=domain_env)
                if domain_env
                else HTTP(api_key=api_key, api_secret=api_secret, testnet=is_testnet)
            )

            # Cierre reduceOnly: lado opuesto a la posición actual.
            side = "Sell" if pos_amt > 0 else "Buy"

            def _quantize(v: float, step: str) -> str:
                dv = Decimal(str(v)).quantize(Decimal(step), rounding=ROUND_DOWN)
                if dv <= 0:
                    dv = Decimal(step)
                return format(dv, "f")

            def _ceil_to_step(value: float, step: str) -> str:
                dv = Decimal(str(value))
                ds = Decimal(step)
                if ds <= 0:
                    return format(dv, "f")
                q = (dv / ds).to_integral_value(rounding=ROUND_UP)
                out = q * ds
                if out <= 0:
                    out = ds
                return format(out, "f")

            def _autocorrect_qty(symbol_: str, qty_s: str) -> str | None:
                try:
                    raw = client.get_instruments_info(category="linear", symbol=str(symbol_).upper())
                    items = raw.get("result", {}).get("list") or []
                    first = items[0] if items else {}
                    lot = first.get("lotSizeFilter") or {}
                    min_qty_s = str(lot.get("minOrderQty") or "")
                    step_s = str(lot.get("qtyStep") or "")
                    if not min_qty_s or not step_s:
                        return None
                    current = float(qty_s)
                    min_qty = float(min_qty_s)
                    target = max(current, min_qty)
                    return _ceil_to_step(target, step_s)
                except Exception:
                    return None

            qty_s = _quantize(qty, "0.001")
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty_s,
                "reduceOnly": True,
            }
            try:
                raw = client.place_order(**params)
            except Exception as exc:
                # Si Bybit rechaza qty/precision, corregir y reintentar una vez.
                msg = str(exc)
                err_code = None
                m = re.search(r"ErrCode:\\s*(\\d+)", msg)
                if m:
                    try:
                        err_code = int(m.group(1))
                    except Exception:
                        err_code = None
                looks_like_qty_error = (
                    (err_code == 10001)
                    or ("minimum limit" in msg.lower())
                    or ("qty" in msg.lower() and "invalid" in msg.lower())
                    or ("precision" in msg.lower())
                )
                if looks_like_qty_error:
                    corrected = _autocorrect_qty(symbol, qty_s)
                    if corrected and corrected != qty_s:
                        print(
                            f"[WATCHER][WARN] Bybit rechazó qty en cierre; reintentando symbol={symbol} "
                            f"qty={qty_s} -> {corrected} err={msg}"
                        )
                        params2 = {**params, "qty": corrected}
                        raw = client.place_order(**params2)
                        params = params2
                    else:
                        raise
                else:
                    raise

            ret_code = raw.get("retCode")
            if ret_code not in (None, 0, "0"):
                msg = raw.get("retMsg") or "BYBIT_ERROR"
                raise RuntimeError(f"Bybit retCode={ret_code} retMsg={msg}")
        else:
            return False
        print(f"[WATCHER][INFO] Cierre reduceOnly MARKET user={user_id} ex={exchange} symbol={symbol} qty={qty} side={side}")
        return True
    except Exception as exc:
        print(f"[WATCHER][WARN] No se pudo cerrar posición user={user_id} ex={exchange} symbol={symbol}: {exc}")
        return False


def _evaluate_thresholds(current_price: float, ts) -> list[dict]:
    """
    Evalúa si el precio actual dispara algún cierre por pérdida/ganancia.
    Devuelve lista de alertas a emitir y ejecuta cierre reduceOnly MARKET cuando corresponde.
    """
    alerts = []
    updated = False
    keep_thresholds = []

    price_cache: dict[tuple[str, str], float | None] = {}
    for th in _thresholds:
        user_id = th.get("user_id")
        exchange = th.get("exchange")
        symbol = th.get("symbol", SYMBOL_DISPLAY.replace(".P", ""))
        direction = th.get("direction")
        signal_direction = th.get("signal_direction") or None
        entry = float(th.get("entry_price") or 0)
        loss_price = float(th.get("loss_price") or 0)
        gain_raw = th.get("gain_price")
        gain_price = float(gain_raw) if gain_raw not in (None, "") else None
        fired_loss = th.get("fired_loss", False)
        fired_gain = th.get("fired_gain", False)
        triggered_kind = th.get("triggered_kind")
        last_attempt = float(th.get("last_close_attempt") or 0.0)
        now_ts = time.time()

        if entry <= 0:
            continue
        # Si ya no hay posición, limpiar registro
        pos_amt = _current_position(user_id, exchange, symbol)
        if pos_amt is None:
            print(
                f"[WATCHER][THRESHOLDS][SKIP] user={user_id} ex={exchange} symbol={symbol} "
                f"reason=position_unknown"
            )
            keep_thresholds.append(th)
            continue
        if pos_amt == 0:
            print(
                f"[WATCHER][THRESHOLDS][CLEAN] user={user_id} ex={exchange} symbol={symbol} "
                f"reason=no_position"
            )
            updated = True
            continue

        used_price = None
        price_key = (str(exchange).lower(), symbol)
        if price_key in price_cache:
            used_price = price_cache[price_key]
        else:
            if exchange:
                ex_l = str(exchange).lower()
                try:
                    manager = _account_manager or _load_manager()
                    if manager is not None and user_id:
                        account = manager.get_account(user_id)
                        cred = account.get_exchange(exchange)
                        if ex_l == "binance":
                            used_price = _binance_mark_price(cred, symbol)
                        elif ex_l == "bybit":
                            used_price = _bybit_mark_price(cred, symbol)
                except Exception:
                    used_price = None
            price_cache[price_key] = used_price
        if used_price is None:
            used_price = current_price
        try:
            used_price = float(used_price)
        except Exception:
            continue
        if used_price <= 0:
            continue

        flip_direction = signal_direction or _opposite_direction(direction)
        if triggered_kind:
            if now_ts - last_attempt < THRESHOLDS_RETRY_SECONDS:
                keep_thresholds.append(th)
                continue
            close_ok = _close_position(user_id, exchange, symbol, direction)
            th["last_close_attempt"] = now_ts
            if close_ok:
                print(
                    f"[WATCHER][THRESHOLDS][CLOSE] user={user_id} ex={exchange} symbol={symbol} "
                    f"ok=True kind={triggered_kind}"
                )
                open_ok, _ = _execute_trade_for_target(
                    user_id,
                    exchange,
                    flip_direction,
                    symbol,
                    used_price,
                    source_event="threshold_flip",
                    signal_direction=signal_direction,
                )
                if not open_ok:
                    print(
                        f"[WATCHER][WARN] No se pudo abrir flip user={user_id} ex={exchange} "
                        f"symbol={symbol} dir={flip_direction}"
                    )
                alerts.append(
                    {
                        "type": "auto_close",
                        "timestamp": ts,
                        "message": (
                            f"{symbol} {STREAM_INTERVAL}\n"
                            f"Cierre {direction.upper()} por {triggered_kind}\n"
                            f"Entrada: {entry:.2f}\n"
                            f"Último: {used_price:.2f}"
                        ),
                        "direction": direction,
                        "user_id": user_id,
                        "exchange": exchange,
                    }
                )
                updated = True
                continue
            print(
                f"[WATCHER][THRESHOLDS][RETRY] user={user_id} ex={exchange} symbol={symbol} "
                f"kind={triggered_kind} next_in={THRESHOLDS_RETRY_SECONDS}s"
            )
            updated = True
            keep_thresholds.append(th)
            continue

        hit_loss = False
        hit_gain = False
        if direction == "long":
            hit_loss = (not fired_loss) and used_price <= loss_price
            hit_gain = gain_price is not None and (not fired_gain) and used_price >= gain_price
        else:  # short
            hit_loss = (not fired_loss) and used_price >= loss_price
            hit_gain = gain_price is not None and (not fired_gain) and used_price <= gain_price

        if hit_loss or hit_gain:
            kind = "ganancia" if hit_gain else f"pérdida -{int(LOSS_PCT * 100)}%"
            # Ejecuta cierre reduceOnly MARKET del tamaño actual
            print(
                f"[WATCHER][THRESHOLDS][TRIGGER] user={user_id} ex={exchange} symbol={symbol} dir={direction} "
                f"last={used_price:.6f} entry={entry:.6f} loss={loss_price:.6f} gain={gain_price} kind={kind}"
            )
            close_ok = _close_position(user_id, exchange, symbol, direction)
            th["last_close_attempt"] = now_ts
            if close_ok:
                print(
                    f"[WATCHER][THRESHOLDS][CLOSE] user={user_id} ex={exchange} symbol={symbol} ok=True kind={kind}"
                )
                open_ok, _ = _execute_trade_for_target(
                    user_id,
                    exchange,
                    flip_direction,
                    symbol,
                    used_price,
                    source_event="threshold_flip",
                    signal_direction=signal_direction,
                )
                if not open_ok:
                    print(
                        f"[WATCHER][WARN] No se pudo abrir flip user={user_id} ex={exchange} "
                        f"symbol={symbol} dir={flip_direction}"
                    )
                alerts.append(
                    {
                        "type": "auto_close",
                        "timestamp": ts,
                        "message": (
                            f"{symbol} {STREAM_INTERVAL}\n"
                            f"Cierre {direction.upper()} por {kind}\n"
                            f"Entrada: {entry:.2f}\n"
                            f"Último: {used_price:.2f}"
                        ),
                        "direction": direction,
                        "user_id": user_id,
                        "exchange": exchange,
                    }
                )
                updated = True
                # una vez disparado, removemos el registro (se reemplaza con la próxima operación)
                continue
            th["triggered_kind"] = kind
            th["fired_loss"] = hit_loss or fired_loss
            th["fired_gain"] = hit_gain or fired_gain
            print(
                f"[WATCHER][THRESHOLDS][RETRY] user={user_id} ex={exchange} symbol={symbol} "
                f"kind={kind} next_in={THRESHOLDS_RETRY_SECONDS}s"
            )
            updated = True
            keep_thresholds.append(th)
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
        if pos_amt is None:
            return False
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
        if pos_amt is None:
            return False
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
        if TRADING_DRY_RUN:
            print(
                f"[WATCHER][INFO] Dry-run activo: se omite cierre real de opuesta user={user_id} ex={exchange} symbol={symbol}"
            )
            return True
        account = _account_manager.get_account(user_id)
        cred = account.get_exchange(exchange)
        pos_amt = _current_position(user_id, exchange, symbol)
        if pos_amt is None:
            print(
                f"[WATCHER][WARN] Posición opuesta desconocida user={user_id} ex={exchange} symbol={symbol}"
            )
            return False
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
        elif exchange.lower() == "bybit":
            from decimal import Decimal, ROUND_DOWN, ROUND_UP
            import re
            from pybit.unified_trading import HTTP  # type: ignore

            api_key, api_secret = cred.resolve_keys(os.environ)
            is_testnet = cred.environment != ExchangeEnvironment.LIVE
            domain_env = os.getenv("BYBIT_DOMAIN_TESTNET" if is_testnet else "BYBIT_DOMAIN")
            client = (
                HTTP(api_key=api_key, api_secret=api_secret, testnet=False, domain=domain_env)
                if domain_env
                else HTTP(api_key=api_key, api_secret=api_secret, testnet=is_testnet)
            )

            side = "Sell" if pos_amt > 0 else "Buy"

            def _quantize(v: float, step: str) -> str:
                dv = Decimal(str(v)).quantize(Decimal(step), rounding=ROUND_DOWN)
                if dv <= 0:
                    dv = Decimal(step)
                return format(dv, "f")

            def _ceil_to_step(value: float, step: str) -> str:
                dv = Decimal(str(value))
                ds = Decimal(step)
                if ds <= 0:
                    return format(dv, "f")
                q = (dv / ds).to_integral_value(rounding=ROUND_UP)
                out = q * ds
                if out <= 0:
                    out = ds
                return format(out, "f")

            def _autocorrect_qty(symbol_: str, qty_s: str) -> str | None:
                try:
                    raw = client.get_instruments_info(category="linear", symbol=str(symbol_).upper())
                    items = raw.get("result", {}).get("list") or []
                    first = items[0] if items else {}
                    lot = first.get("lotSizeFilter") or {}
                    min_qty_s = str(lot.get("minOrderQty") or "")
                    step_s = str(lot.get("qtyStep") or "")
                    if not min_qty_s or not step_s:
                        return None
                    current = float(qty_s)
                    min_qty = float(min_qty_s)
                    target = max(current, min_qty)
                    return _ceil_to_step(target, step_s)
                except Exception:
                    return None

            qty_s = _quantize(qty, "0.001")
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty_s,
                "reduceOnly": True,
            }
            try:
                raw = client.place_order(**params)
            except Exception as exc:
                msg = str(exc)
                err_code = None
                m = re.search(r"ErrCode:\\s*(\\d+)", msg)
                if m:
                    try:
                        err_code = int(m.group(1))
                    except Exception:
                        err_code = None
                looks_like_qty_error = (
                    (err_code == 10001)
                    or ("minimum limit" in msg.lower())
                    or ("qty" in msg.lower() and "invalid" in msg.lower())
                    or ("precision" in msg.lower())
                )
                if looks_like_qty_error:
                    corrected = _autocorrect_qty(symbol, qty_s)
                    if corrected and corrected != qty_s:
                        print(
                            f"[WATCHER][WARN] Bybit rechazó qty en cierre; reintentando symbol={symbol} "
                            f"qty={qty_s} -> {corrected} err={msg}"
                        )
                        params2 = {**params, "qty": corrected}
                        raw = client.place_order(**params2)
                        params = params2
                    else:
                        raise
                else:
                    raise

            ret_code = raw.get("retCode")
            if ret_code not in (None, 0, "0"):
                msg = raw.get("retMsg") or "BYBIT_ERROR"
                raise RuntimeError(f"Bybit retCode={ret_code} retMsg={msg}")
        else:
            return False
        print(
            f"[WATCHER][INFO] Cierre reduceOnly (MARKET) de posición opuesta qty={qty} side={side} en {symbol} ex={exchange}"
        )
        if CLOSE_OPPOSITE_TIMEOUT_SECONDS > 0:
            print(
                f"[WATCHER][INFO] Esperando cierre completo de posición opuesta user={user_id} ex={exchange} symbol={symbol}"
            )
            deadline = time.time() + CLOSE_OPPOSITE_TIMEOUT_SECONDS
            while True:
                pos_now = _current_position(user_id, exchange, symbol)
                if pos_now is None:
                    print(
                        f"[WATCHER][WARN] Posición opuesta desconocida durante espera "
                        f"user={user_id} ex={exchange} symbol={symbol}"
                    )
                    return False
                if abs(pos_now) < 1e-8:
                    break
                if time.time() >= deadline:
                    print(
                        f"[WATCHER][WARN] Posición opuesta sigue abierta tras espera "
                        f"user={user_id} ex={exchange} symbol={symbol} pos={pos_now}"
                    )
                    return False
                time.sleep(max(CLOSE_OPPOSITE_POLL_SECONDS, 0.1))
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
        signal_direction = (event.get("direction") or "").lower()
        order_direction = _opposite_direction(signal_direction)
        side = _direction_to_side(order_direction)
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
    for user_id, exchange in targets:
        try:
            account = _account_manager.get_account(user_id) if _account_manager else None
            cred = account.get_exchange(exchange) if account else None
            notional = None
            if cred:
                notional = cred.notional_usdt
        except Exception:
            notional = None
        try:
            quantity = _resolve_quantity({**event, "price": price}, notional_usdt=notional)
        except Exception as exc:
            print(f"[WATCHER][WARN] Cantidad inválida para trading ({exc}) usuario={user_id} exchange={exchange}")
            continue
        if exchange.lower() == "binance":
            # Evita quedar por debajo del mínimo notional tras el redondeo de Binance.
            step = 0.001
            quantity = math.ceil(quantity / step) * step
        symbol = event.get("symbol") or SYMBOL_DISPLAY.replace(".P", "")
        if cred and cred.extra:
            symbol = cred.extra.get("symbol", symbol)

        pos_amt = _current_position(user_id, exchange, symbol)
        pos_dir = None
        if pos_amt is not None:
            if pos_amt > 0:
                pos_dir = "long"
            elif pos_amt < 0:
                pos_dir = "short"
        if pos_dir == signal_direction:
            _update_threshold_from_signal(
                user_id,
                exchange,
                symbol,
                position_direction=pos_dir,
                signal_direction=signal_direction,
                entry_price=price,
            )
            print(
                f"[WATCHER][INFO] Señal coincide con posición {pos_dir}; se actualiza threshold y no se abre orden."
            )
            continue

        # Nota: no hay overrides por env para el monto. El sizing sale del YAML (DashCRUD) por usuario/exchange.
        had_opposite = _has_opposite_position(user_id, exchange, order_direction, symbol)
        # Si hay posición opuesta, envía cierre reduceOnly y entrada simultánea en el mismo precio.
        # Se mantienen TP/SL previos hasta que el cierre se ejecute.
        if had_opposite:
            if not _close_opposite_position(user_id, exchange, order_direction, symbol, price):
                print(f"[WATCHER][WARN] No se pudo cerrar posición opuesta en {symbol}; se omite señal.")
                continue
        key_dir = (user_id, exchange)
        last_dir = _last_order_direction.get(key_dir)
        if order_direction and order_direction == (last_dir or "").lower():
            print(f"[WATCHER][INFO] Orden {order_direction} ya colocada en {exchange}; se ignora señal.")
            continue
        if _has_open_position_same_direction(user_id, exchange, order_direction, symbol):
            print(f"[WATCHER][INFO] Ya hay posición {order_direction} abierta en {symbol}; se omite señal.")
            continue

        extra = {
            "source_event": event.get("type", "unknown"),
            "event_timestamp": str(event.get("timestamp")),
            "account": user_id,
            "exchange": exchange,
            "signal_direction": signal_direction,
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
                _last_order_direction[key_dir] = order_direction
                # Registra umbral (-2%) para contratrend
                entry_used = float(response.avg_price or price)
                _register_threshold(user_id, exchange, symbol, order_direction, entry_used, signal_direction)
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
THRESHOLDS_DUMP_SECONDS = float(os.getenv("THRESHOLDS_DUMP_SECONDS", "300"))
THRESHOLDS_RETRY_SECONDS = float(os.getenv("THRESHOLDS_RETRY_SECONDS", "10"))
THRESHOLDS_DUMP_SECONDS = float(os.getenv("THRESHOLDS_DUMP_SECONDS", "300"))


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


def _dump_thresholds(ts: datetime) -> None:
    manager = _account_manager or _load_manager()
    print(f"[WATCHER][THRESHOLDS][DUMP] ts={ts.isoformat()} count={len(_thresholds)}")
    if not _thresholds:
        return
    price_cache: dict[tuple[str, str], float | None] = {}
    for th in _thresholds:
        user_id = th.get("user_id")
        exchange = th.get("exchange")
        symbol = th.get("symbol", SYMBOL_DISPLAY.replace(".P", ""))
        signal_direction = th.get("signal_direction")
        entry = float(th.get("entry_price") or 0)
        loss_price = float(th.get("loss_price") or 0)
        gain_raw = th.get("gain_price")
        gain_price = float(gain_raw) if gain_raw not in (None, "") else None
        mark = None
        triggered_kind = th.get("triggered_kind")
        last_attempt = th.get("last_close_attempt")
        ex_key = str(exchange).lower() if exchange else ""
        cache_key = (ex_key, symbol)
        if cache_key in price_cache:
            mark = price_cache[cache_key]
        else:
            if manager is not None and user_id and exchange:
                try:
                    account = manager.get_account(user_id)
                    cred = account.get_exchange(exchange)
                    if ex_key == "binance":
                        mark = _binance_mark_price(cred, symbol)
                    elif ex_key == "bybit":
                        mark = _bybit_mark_price(cred, symbol)
                except Exception:
                    mark = None
            price_cache[cache_key] = mark
        print(
            f"[WATCHER][THRESHOLDS][DUMP] user={user_id} ex={exchange} symbol={symbol} "
            f"entry={entry:.6f} loss={loss_price:.6f} gain={gain_price} mark={mark} "
            f"signal_dir={signal_direction} "
            f"triggered={triggered_kind} last_attempt={last_attempt}"
        )


def main():
    seen = []
    _notify_startup()
    last_threshold_check = 0.0
    last_disabled_check = 0.0
    last_threshold_dump = 0.0
    if THRESHOLDS_CLEAR_ON_STARTUP:
        _clear_thresholds_file()
    if THRESHOLDS_REBUILD_ON_STARTUP:
        try:
            _rebuild_thresholds_from_open_positions()
        except Exception as exc:
            print(f"[WATCHER][WARN] Reconstrucción de umbrales falló: {exc}")
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
                ts_eval = datetime.now(timezone.utc)
                try:
                    extra_alerts = _evaluate_thresholds(current_price or 0.0, ts_eval)
                    if extra_alerts:
                        send_alerts(extra_alerts)
                except Exception as exc:
                    print(f"[ALERT][WARN] Falló evaluación periódica de umbrales ({exc})")
            except Exception:
                pass

        if THRESHOLDS_DUMP_SECONDS > 0 and now_ts - last_threshold_dump >= THRESHOLDS_DUMP_SECONDS:
            last_threshold_dump = now_ts
            try:
                _dump_thresholds(datetime.now(timezone.utc))
            except Exception as exc:
                print(f"[WATCHER][WARN] Falló dump de umbrales ({exc})")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
