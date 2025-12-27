# CONTEXT.md (estrategia para subagentes)

Este archivo está pensado para *cargar contexto* de forma estable y reproducible (para LLMs/subagentes) sin depender de exploración ad‑hoc del repo.

## Objetivo
- Dar una vista rápida de *qué hace el repo* y *dónde vive el código importante*.
- Definir un orden recomendado de lectura (y qué ignorar).
- Incluir el **código completo** de los archivos primordiales (sin secretos).

## Qué NO incluir
- Archivos generados/volátiles: `*.csv`, `__pycache__/`, `.venv/`, outputs de backtests.
- Secretos: `.env` real, llaves API, tokens, seeds.

## Orden recomendado de lectura (subagente)
1. `requirements.txt` (dependencias)
2. `trading/` (librería base: cuentas, exchanges, órdenes, logging)
3. `watcher_alertas.py` + `alerts.py` + `velas.py` (flujo principal de velas/alertas)
4. `telegram_bot_commands.py` + `check_telegram.py` (integración Telegram)
5. `backtest/` (motor de backtesting y dashboards)
6. `scripts/` (utilidades puntuales)

## Regeneración (cuando cambie el repo)
- Recomendado: regenerar este archivo automáticamente (y volver a revisar que no se haya filtrado ningún secreto).
- Comando sugerido (adaptar la lista de archivos “primordiales” según tu necesidad):

```bash
python - <<'PY'
from pathlib import Path
files = [
  "requirements.txt",
  "watcher_alertas.py",
  "alerts.py",
  "velas.py",
  "telegram_bot_commands.py",
  "trading/__init__.py",
]
out = ["# CONTEXT.md\\n", "## Código\\n"]
for f in files:
  p = Path(f)
  out.append(f"### `{f}`\\n\\n```\\n{p.read_text(encoding='utf-8', errors='replace').rstrip()}\\n```\\n")
Path("CONTEXT.md").write_text("\\n".join(out), encoding="utf-8")
PY
```

## Variables de entorno (plantilla)
Este repo usa un `.env` local (no versionar). Para entrenamiento/CI usar una plantilla con placeholders.

- `SYMBOL`, `TZ`, `CHANNEL_INTERVAL`, `STREAM_INTERVAL`, `LIMIT_CHANNEL`, `LIMIT_STREAM`
- `RB_MULTI`, `RB_INIT_BAR`, `SLEEP_FALLBACK`, `WARN_TOO_MUCH`, `TABLE_CSV_PATH`
- `BINANCE_UM_BASE_URL`, `BASE_URL`
- `PLOT_STREAM_BARS`, `PLOT_CHANNEL_BARS`, `PAGINATE_PAGE_LIMIT`, `PAGE_SLEEP_SEC`
- `BB_LENGTH`, `BB_MULT`, `BB_DIRECTION`, `BB_STD_DDOF`, `BB_LINE_WIDTH`, `BB_*_COLOR`, `BB_FILL_ALPHA`
- `ALERT_*` (polling, tolerancias, buffers), `ALERT_ENABLE_BOLLINGER_SIGNALS`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_IDS` (NO incluir valores reales)
- `STRAT_*_PATH` (paths de output)
- `BACKTEST_*`
- `DYDX_*` / `*_DYDX_*` (NO incluir valores reales)

## Estructura principal
- `trading/`: librería reusable (cuentas/exchanges/órdenes/logging)
- `watcher_alertas.py`: watcher principal (orquestación de alertas)
- `alerts.py`: lógica de alertas (stream/canales/señales)
- `velas.py`: obtención/procesamiento de velas
- `telegram_bot_commands.py`: comandos/handlers del bot
- `backtest/`: runner + generación de dashboard
- `scripts/`: validaciones/tests puntuales
- `docs/`: guías (DYDX/OCI/etc.)

---

## Código (archivos primordiales)

### `requirements.txt`

```text
anyio==4.12.0
asn1crypto==1.5.1
bech32==1.2.0
binance-futures-connector==4.1.0
bip_utils==2.10.0
cbor2==5.7.1
certifi==2025.11.12
cffi==2.0.0
charset-normalizer==3.4.4
coincurve==20.0.0
contourpy==1.3.2
crcmod==1.7
cycler==0.12.1
dydx-v4-client==1.1.5
ecdsa==0.19.1
ed25519-blake2b==1.4.1
exceptiongroup==1.3.1
fonttools==4.60.1
grpcio==1.76.0
grpcio-tools==1.76.0
h11==0.16.0
httpcore==1.0.9
httpx==0.27.2
idna==3.11
kiwisolver==1.4.9
matplotlib==3.10.7
mplfinance==0.12.10b0
narwhals==2.11.0
numpy==2.2.6
packaging==25.0
pandas==2.2.3
pillow==12.0.0
plotly==6.4.0
protobuf==6.33.2
py-sr25519-bindings==0.2.3
pycparser==2.23
pycryptodome==3.23.0
PyNaCl==1.6.1
pyparsing==3.2.5
python-dateutil==2.9.0.post0
python-dotenv==1.2.1
pytz==2025.2
PyYAML==6.0.3
requests==2.32.3
six==1.17.0
sniffio==1.3.1
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
v4-proto==8.2.0
websocket-client==1.9.0
webcolors==25.10.0
pybit==5.13.0
```

### `trading/__init__.py`

```python
"""
Paquete de soporte para ejecución real y gestión de cuentas.

Se divide en submódulos:
- trading.exchanges: abstracciones y clientes específicos de cada exchange.
- trading.accounts: modelos y utilitarios para manejar credenciales multiusuario.
- trading.orders: estructuras comunes para órdenes, posiciones y resultados.
- trading.utils: utilitarios compartidos (logging, tiempo, etc.).
"""

from .orders.models import OrderRequest, OrderResponse, OrderSide, OrderType, TimeInForce
from .accounts.models import AccountConfig, ExchangeCredential, ExchangeEnvironment

__all__ = [
    "OrderRequest",
    "OrderResponse",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "AccountConfig",
    "ExchangeCredential",
    "ExchangeEnvironment",
]
```

### `trading/utils/__init__.py`

```python
"""
Utilitarios compartidos para logging, fechas, etc.
"""
```

### `trading/utils/logging.py`

```python
import logging
import os


def get_logger(name: str) -> logging.Logger:
    """
    Devuelve un logger configurado con nivel INFO por defecto.
    Se respeta la variable de entorno TRADING_LOG_LEVEL si está definida.
    """
    level_name = os.getenv("TRADING_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    else:
        logger.setLevel(level)
    return logger
```

### `trading/accounts/__init__.py`

```python
"""
Gestión de cuentas y credenciales multiusuario.
"""

from .models import AccountConfig, ExchangeCredential, ExchangeEnvironment
from .manager import AccountManager

__all__ = [
    "AccountConfig",
    "ExchangeCredential",
    "ExchangeEnvironment",
    "AccountManager",
]
```

### `trading/accounts/models.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping


class ExchangeEnvironment(str, Enum):
    TESTNET = "testnet"
    LIVE = "live"


@dataclass(slots=True)
class ExchangeCredential:
    exchange: str
    api_key_env: str
    api_secret_env: str
    environment: ExchangeEnvironment = ExchangeEnvironment.TESTNET
    notional_usdt: float | None = None
    leverage: int | None = None
    extra: Dict[str, Any] = field(default_factory=dict)
    # dYdX v4 soporte: passphrase/stark keys y notional USDC / max position
    passphrase_env: str | None = None
    stark_key_env: str | None = None
    notional_usdc: float | None = None
    max_position_usdc: float | None = None
    margin_mode: str | None = None  # 'isolated' o 'cross' (dYdX soporta isolated markets)

    def resolve_keys(self, env: Mapping[str, str]) -> tuple[str, str]:
        api_key = env.get(self.api_key_env)
        api_secret = env.get(self.api_secret_env)
        if not api_key or not api_secret:
            raise RuntimeError(
                f"Credenciales faltantes para {self.exchange}: "
                f"{self.api_key_env}/{self.api_secret_env}"
            )
        return api_key, api_secret

    def resolve_optional(self, env: Mapping[str, str], key: str | None) -> str | None:
        if key is None:
            return None
        return env.get(key) or None


@dataclass(slots=True)
class AccountConfig:
    user_id: str
    label: str
    enabled: bool = True
    exchanges: Dict[str, ExchangeCredential] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_exchange(self, exchange_name: str) -> ExchangeCredential:
        key = exchange_name.lower()
        if key not in self.exchanges:
            raise KeyError(f"La cuenta {self.user_id} no tiene credenciales para {exchange_name}.")
        return self.exchanges[key]
```

### `trading/accounts/manager.py`

```python
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .models import AccountConfig, ExchangeCredential, ExchangeEnvironment

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


class AccountManager:
    """
    Administra múltiples cuentas/usuarios y sus credenciales por exchange.
    """

    def __init__(self, accounts: Iterable[AccountConfig]):
        self._accounts: Dict[str, AccountConfig] = {}
        for account in accounts:
            self._accounts[account.user_id] = account

    @classmethod
    def empty(cls) -> "AccountManager":
        return cls(accounts=[])

    @classmethod
    def from_dict(cls, data: Mapping) -> "AccountManager":
        users = data.get("users") or data.get("accounts") or []
        accounts: list[AccountConfig] = []
        for entry in users:
            user_id = entry.get("id") or entry.get("user_id")
            if not user_id:
                raise ValueError("Cada cuenta debe especificar 'id'.")
            label = entry.get("label") or user_id
            exchanges_data = entry.get("exchanges") or {}
            exchanges: Dict[str, ExchangeCredential] = {}
            for ex_name, ex_conf in exchanges_data.items():
                exchange = ex_conf.get("exchange", ex_name).lower()
                env_value = (ex_conf.get("environment") or ExchangeEnvironment.TESTNET.value).lower()
                environment = ExchangeEnvironment(env_value)
                notional_val = ex_conf.get("notional_usdt")
                notional = float(notional_val) if notional_val not in (None, "") else None
                leverage_val = ex_conf.get("leverage")
                leverage = int(leverage_val) if leverage_val not in (None, "") else None
                cred = ExchangeCredential(
                    exchange=exchange,
                    api_key_env=ex_conf["api_key_env"],
                    api_secret_env=ex_conf["api_secret_env"],
                    environment=environment,
                    notional_usdt=notional,
                    leverage=leverage,
                    passphrase_env=ex_conf.get("passphrase_env"),
                    stark_key_env=ex_conf.get("stark_key_env"),
                    notional_usdc=float(ex_conf["notional_usdc"]) if ex_conf.get("notional_usdc") not in (None, "") else None,
                    max_position_usdc=float(ex_conf["max_position_usdc"]) if ex_conf.get("max_position_usdc") not in (None, "") else None,
                    margin_mode=ex_conf.get("margin_mode"),
                    extra={
                        k: v
                        for k, v in ex_conf.items()
                        if k
                        not in {
                            "api_key_env",
                            "api_secret_env",
                            "environment",
                            "notional_usdt",
                            "leverage",
                            "passphrase_env",
                            "stark_key_env",
                            "notional_usdc",
                            "max_position_usdc",
                            "margin_mode",
                        }
                    },
                )
                exchanges[exchange] = cred
            metadata = entry.get("metadata") or {}
            accounts.append(
                AccountConfig(
                    user_id=user_id,
                    label=label,
                    enabled=bool(entry.get("enabled", True)),
                    exchanges=exchanges,
                    metadata=metadata,
                )
            )
        return cls(accounts)

    @classmethod
    def from_file(cls, path: Path) -> "AccountManager":
        if not path.exists():
            raise FileNotFoundError(f"No se encontró archivo de cuentas: {path}")

        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML no está instalado. `pip install pyyaml` para leer archivos YAML.")
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        else:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        return cls.from_dict(data)

    def list_accounts(self) -> list[AccountConfig]:
        return list(self._accounts.values())

    def get_account(self, user_id: str) -> AccountConfig:
        try:
            return self._accounts[user_id]
        except KeyError as exc:
            raise KeyError(f"No existe la cuenta '{user_id}'.") from exc

    def get_exchange_credential(self, user_id: str, exchange: str) -> ExchangeCredential:
        account = self.get_account(user_id)
        return account.get_exchange(exchange)

    def resolve_keys(self, user_id: str, exchange: str, env: Optional[Mapping[str, str]] = None) -> tuple[str, str]:
        credential = self.get_exchange_credential(user_id, exchange)
        env_mapping = env or os.environ
        return credential.resolve_keys(env_mapping)

    def to_dict(self) -> dict:
        """
        Devuelve una representación serializable a JSON/YAML.

        Se normaliza `environment` a su valor string para evitar que se
        serialice como Enum en el archivo de cuentas.
        """

        def _serialize_credential(cred: ExchangeCredential) -> dict:
            data = asdict(cred)
            data["environment"] = cred.environment.value
            return data

        def _serialize_account(acc: AccountConfig) -> dict[str, Any]:
            return {
                "id": acc.user_id,
                "label": acc.label,
                "enabled": acc.enabled,
                "metadata": acc.metadata or {},
                "exchanges": {name: _serialize_credential(cred) for name, cred in acc.exchanges.items()},
            }

        return {"users": [_serialize_account(acc) for acc in self._accounts.values()]}

    # --- Mutadores -----------------------------------------------------

    def upsert_account(
        self,
        user_id: str,
        *,
        label: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        enabled: Optional[bool] = None,
    ) -> AccountConfig:
        """
        Crea o actualiza una cuenta. No cambia el user_id existente.
        """
        if not user_id:
            raise ValueError("user_id no puede ser vacío.")
        account = self._accounts.get(user_id)
        if account:
            if label:
                account.label = label
            if metadata is not None:
                account.metadata = dict(metadata)
            if enabled is not None:
                account.enabled = bool(enabled)
            return account

        new_account = AccountConfig(
            user_id=user_id,
            label=label or user_id,
            enabled=True if enabled is None else bool(enabled),
            exchanges={},
            metadata=dict(metadata or {}),
        )
        self._accounts[user_id] = new_account
        return new_account

    def remove_account(self, user_id: str) -> None:
        try:
            del self._accounts[user_id]
        except KeyError as exc:
            raise KeyError(f"No existe la cuenta '{user_id}'.") from exc

    def rename_account(self, old_id: str, new_id: str) -> AccountConfig:
        if not new_id:
            raise ValueError("new_id no puede ser vacío.")
        if new_id == old_id:
            return self.get_account(old_id)
        if new_id in self._accounts:
            raise ValueError(f"Ya existe la cuenta '{new_id}'.")
        account = self.get_account(old_id)
        del self._accounts[old_id]
        account.user_id = new_id
        self._accounts[new_id] = account
        return account

    def upsert_exchange(self, user_id: str, credential: ExchangeCredential) -> None:
        account = self.get_account(user_id)
        credential.exchange = credential.exchange.lower()
        account.exchanges[credential.exchange] = credential

    def remove_exchange(self, user_id: str, exchange: str) -> None:
        account = self.get_account(user_id)
        key = exchange.lower()
        if key not in account.exchanges:
            raise KeyError(f"La cuenta {user_id} no tiene credenciales para {exchange}.")
        del account.exchanges[key]

    def save_to_file(self, path: Path) -> None:
        """
        Persiste las cuentas al archivo indicado (YAML o JSON).
        """
        data = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML no está instalado. `pip install pyyaml` para escribir archivos YAML.")
            with path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=False)
            return

        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
```

### `trading/exchanges/__init__.py`

```python
"""
Registro de exchanges soportados y sus clientes.
"""

from .base import ExchangeClient, ExchangeRegistry

from . import binance  # noqa: F401  # Registro de cliente Binance por defecto
try:
    from . import dydx  # noqa: F401  # Registro de cliente dYdX
except Exception:
    dydx = None  # dYdX es opcional; si falta dependencias no bloquea el resto
try:
    from . import bybit  # noqa: F401  # Registro de cliente Bybit
except Exception:
    bybit = None

__all__ = [
    "ExchangeClient",
    "ExchangeRegistry",
]
```

### `trading/exchanges/base.py`

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

from ..accounts.models import AccountConfig, ExchangeCredential
from ..orders.models import CancelRequest, CancelResponse, OrderRequest, OrderResponse


class ExchangeClient(ABC):
    """
    Interface que deben implementar todos los exchanges soportados.
    """

    name: str

    @abstractmethod
    def place_order(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
        order: OrderRequest,
        *,
        dry_run: bool = False,
    ) -> OrderResponse:
        ...

    @abstractmethod
    def cancel_order(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
        request: CancelRequest,
        *,
        dry_run: bool = False,
    ) -> CancelResponse:
        ...

    @abstractmethod
    def fetch_account_balance(self, account: AccountConfig, credential: ExchangeCredential) -> Dict[str, float]:
        ...


class ExchangeRegistry:
    """
    Registro global para mapear nombres de exchanges a implementaciones.
    """

    _clients: Dict[str, Type[ExchangeClient]] = {}

    @classmethod
    def register(cls, client_cls: Type[ExchangeClient]) -> None:
        key = client_cls.name.lower()
        cls._clients[key] = client_cls

    @classmethod
    def get(cls, name: str) -> Type[ExchangeClient]:
        key = name.lower()
        if key not in cls._clients:
            raise KeyError(f"Exchange '{name}' no registrado.")
        return cls._clients[key]

    @classmethod
    def list_names(cls) -> list[str]:
        return sorted(cls._clients.keys())
```

### `trading/exchanges/binance.py`

```python
from __future__ import annotations

import os
import math
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional, List

from binance.um_futures import UMFutures

from .base import ExchangeClient, ExchangeRegistry
from ..accounts.models import AccountConfig, ExchangeCredential, ExchangeEnvironment
from ..orders.models import CancelRequest, CancelResponse, OrderRequest, OrderResponse
from ..utils.logging import get_logger

logger = get_logger("trading.exchanges.binance")


class BinanceClient(ExchangeClient):
    name = "binance"

    def _build_client(self, credential: ExchangeCredential) -> UMFutures:
        api_key, api_secret = credential.resolve_keys(os.environ)
        base_url = "https://testnet.binancefuture.com" if credential.environment == ExchangeEnvironment.TESTNET else None
        if base_url:
            return UMFutures(key=api_key, secret=api_secret, base_url=base_url)
        return UMFutures(key=api_key, secret=api_secret)

    @staticmethod
    def _format_order_params(order: OrderRequest) -> Dict[str, Optional[str]]:
        # Binance UM ETHUSDT: qty step 0.001, tick size 0.1 (ajusta si usas otro símbolo).
        def _quantize(value: float, step: str) -> str:
            dv = Decimal(str(value)).quantize(Decimal(step), rounding=ROUND_DOWN)
            if dv <= 0:
                dv = Decimal(step)
            # Normaliza sin notación científica
            return format(dv, "f")

        params: Dict[str, Optional[str]] = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.type.value,
            "quantity": _quantize(order.quantity, "0.001"),
            "reduceOnly": "true" if order.reduce_only else "false",
        }
        if order.type.value == "MARKET":
            # Binance rechaza timeInForce/isPostOnly en órdenes MARKET
            return params

        # LIMIT / STOP_LIMIT conservan post-only + timeInForce/price
        if not order.reduce_only:
            params["isPostOnly"] = "true"
        if order.time_in_force:
            params["timeInForce"] = order.time_in_force.value
        if order.price:
            params["price"] = _quantize(order.price, "0.1")
        return params

    def _place_bracket(
        self,
        client: UMFutures,
        symbol: str,
        side: str,
        quantity: float,
        tp: float | None,
        sl: float | None,
    ) -> Dict[str, Any]:
        """
        Envía TP/SL como órdenes condicionadas de mercado con closePosition=true
        (equivalente a reduceOnly) disparadas por MARK_PRICE.
        """
        results: Dict[str, Any] = {}

        def _quant(v: float, step: str) -> str:
            dv = Decimal(str(v)).quantize(Decimal(step), rounding=ROUND_DOWN)
            if dv <= 0:
                dv = Decimal(step)
            return format(dv, "f")

        qty_str = _quant(quantity, "0.001")
        if tp and tp > 0:
            try:
                resp_tp = client.new_order(
                    symbol=symbol,
                    side="SELL" if side == "BUY" else "BUY",
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=_quant(tp, "0.1"),
                    workingType="MARK_PRICE",
                    # closePosition hace que no abra posición nueva y cierre todo
                    closePosition="true",
                    timeInForce="GTC",
                )
                results["tp"] = resp_tp
            except Exception as exc:  # pragma: no cover - externo
                logger.error("Error enviando TP reduceOnly: %s", exc)
                results["tp_error"] = str(exc)

        if sl and sl > 0:
            try:
                resp_sl = client.new_order(
                    symbol=symbol,
                    side="SELL" if side == "BUY" else "BUY",
                    type="STOP_MARKET",
                    stopPrice=_quant(sl, "0.1"),
                    workingType="MARK_PRICE",
                    closePosition="true",
                    timeInForce="GTC",
                )
                results["sl"] = resp_sl
            except Exception as exc:  # pragma: no cover - externo
                logger.error("Error enviando SL reduceOnly: %s", exc)
                results["sl_error"] = str(exc)

        return results

    def _current_position_qty(self, client: UMFutures, symbol: str) -> float:
        """
        Devuelve el tamaño de posición actual (signed: >0 long, <0 short) para el símbolo.
        Usa get_position_risk, que está soportado en la lib actual.
        """
        try:
            positions = client.get_position_risk(symbol=symbol)
            if positions:
                pos_amt = positions[0].get("positionAmt")
                return float(pos_amt or 0.0)
        except Exception as exc:  # pragma: no cover - externo
            logger.error("No se pudo obtener posición actual para %s: %s", symbol, exc)
        return 0.0

    def _cancel_open_reduce_only(self, client: UMFutures, symbol: str) -> List[Dict[str, Any]]:
        """
        Cancela órdenes abiertas reduceOnly del símbolo (TP/SL previos) antes de colocar nuevos.
        """
        canceled: List[Dict[str, Any]] = []
        try:
            open_orders = client.get_open_orders(symbol=symbol)
        except Exception as exc:  # pragma: no cover - externo
            logger.error("Error listando órdenes abiertas para cancelar reduceOnly: %s", exc)
            return canceled

        for order in open_orders:
            try:
                if not bool(order.get("reduceOnly")):
                    continue
                oid = order.get("orderId")
                resp = client.cancel_order(symbol=symbol, orderId=oid)
                canceled.append(resp)
            except Exception as exc:  # pragma: no cover - externo
                logger.error("Error cancelando orden reduceOnly %s: %s", order.get("orderId"), exc)
        return canceled

    def place_order(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
        order: OrderRequest,
        *,
        dry_run: bool = False,
    ) -> OrderResponse:
        order.validate()

        logger.info(
            "Procesando orden (dry_run=%s) usuario=%s exchange=%s symbol=%s side=%s qty=%s type=%s price=%s",
            dry_run,
            account.user_id,
            credential.exchange,
            order.symbol,
            order.side.value,
            order.quantity,
            order.type.value,
            order.price,
        )

        if dry_run:
            return OrderResponse(
                success=True,
                status="SIMULATED",
                exchange_order_id=None,
                filled_quantity=order.quantity,
                avg_price=order.price,
                raw={
                    "dry_run": True,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": order.quantity,
                    "type": order.type.value,
                    "price": order.price,
                    "reduce_only": order.reduce_only,
                    "time_in_force": order.time_in_force.value,
                },
            )

        try:
            client = self._build_client(credential)
            params = self._format_order_params(order)
            response = client.new_order(**params)
            status = response.get("status") or "NEW"
            order_id = str(response.get("orderId") or "")
            filled_qty = float(response.get("executedQty") or 0.0)
            avg_price = float(response.get("avgPrice") or order.price or 0.0)
            tp = order.extra_params.get("tp") if order.extra_params else None
            sl = order.extra_params.get("sl") if order.extra_params else None
            # Binance rechaza TP/SL con el endpoint estándar (error -4120). Deshabilitamos brackets
            # y delegamos los cierres al watcher (cierres MARKET por ±5%/±9%).
            bracket_raw: Dict[str, Any] = {}
            logger.info(
                "Orden enviada (entry + bracket) symbol=%s side=%s qty=%s tp=%s sl=%s bracket=%s",
                order.symbol,
                order.side.value,
                order.quantity,
                tp,
                sl,
                bracket_raw,
            )
            return OrderResponse(
                success=True,
                status=status,
                exchange_order_id=order_id,
                filled_quantity=filled_qty,
                avg_price=avg_price,
                raw={"entry": response, "bracket": bracket_raw},
            )
        except Exception as exc:
            logger.exception("Error enviando orden a Binance: %s", exc)
            return OrderResponse(success=False, status="ERROR", error=str(exc))

    def cancel_order(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
        request: CancelRequest,
        *,
        dry_run: bool = False,
    ) -> CancelResponse:
        request.validate()
        logger.info(
            "Cancelación (dry_run=%s) usuario=%s exchange=%s symbol=%s order_id=%s client_id=%s",
            dry_run,
            account.user_id,
            credential.exchange,
            request.symbol,
            request.exchange_order_id,
            request.client_order_id,
        )
        return CancelResponse(
            success=True,
            raw={
                "dry_run": dry_run or credential.environment == ExchangeEnvironment.TESTNET,
                "symbol": request.symbol,
                "order_id": request.exchange_order_id,
                "client_order_id": request.client_order_id,
            },
        )

    def fetch_account_balance(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
    ) -> Dict[str, float]:
        logger.info(
            "Consulta de balance (simulado) usuario=%s exchange=%s",
            account.user_id,
            credential.exchange,
        )
        return {"USDT": 0.0}


ExchangeRegistry.register(BinanceClient)
```

### `trading/exchanges/bybit.py`

```python
from __future__ import annotations

import os
from typing import Any, Dict, Optional, List
from decimal import Decimal, ROUND_DOWN

from pybit.unified_trading import HTTP

from .base import ExchangeClient, ExchangeRegistry
from ..accounts.models import AccountConfig, ExchangeCredential, ExchangeEnvironment
from ..orders.models import OrderRequest, OrderResponse, CancelRequest, CancelResponse, OrderSide, OrderType
from ..utils.logging import get_logger

logger = get_logger("trading.exchanges.bybit")


class BybitClient(ExchangeClient):
    name = "bybit"

    def _build_client(self, credential: ExchangeCredential):
        api_key, api_secret = credential.resolve_keys(os.environ)
        is_testnet = credential.environment != ExchangeEnvironment.LIVE
        # pybit v5 unified trading; allow optional custom domain.
        domain_env = os.getenv("BYBIT_DOMAIN_TESTNET" if is_testnet else "BYBIT_DOMAIN")
        if domain_env:
            return HTTP(api_key=api_key, api_secret=api_secret, testnet=False, domain=domain_env)
        return HTTP(api_key=api_key, api_secret=api_secret, testnet=is_testnet)

    @staticmethod
    def _quantize(value: float, step: str) -> str:
        dv = Decimal(str(value)).quantize(Decimal(step), rounding=ROUND_DOWN)
        if dv <= 0:
            dv = Decimal(step)
        return format(dv, "f")

    def _format_order_params(self, order: OrderRequest) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "category": "linear",  # USDT Perp
            "symbol": order.symbol,
            "side": "Buy" if order.side == OrderSide.BUY else "Sell",
            "orderType": "Market" if order.type == OrderType.MARKET else "Limit",
            "qty": self._quantize(order.quantity, "0.001"),
            "reduceOnly": order.reduce_only,
        }
        if order.type == OrderType.LIMIT and order.price:
            params["price"] = self._quantize(order.price, "0.1")
            params["timeInForce"] = "GTC"
        return params

    def place_order(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
        order: OrderRequest,
        *,
        dry_run: bool = False,
    ) -> OrderResponse:
        order.validate()
        logger.info(
            "Procesando orden (dry_run=%s) usuario=%s exchange=%s symbol=%s side=%s qty=%s type=%s price=%s",
            dry_run,
            account.user_id,
            credential.exchange,
            order.symbol,
            order.side.value,
            order.quantity,
            order.type.value,
            order.price,
        )
        if dry_run:
            return OrderResponse(
                success=True,
                status="SIMULATED",
                exchange_order_id=None,
                filled_quantity=order.quantity,
                avg_price=order.price,
                raw={"dry_run": True},
            )
        try:
            client = self._build_client(credential)
            params = self._format_order_params(order)
            raw = client.place_order(**params)
            order_id = str(raw.get("result", {}).get("orderId") or "")
            status = raw.get("result", {}).get("orderStatus") or raw.get("retMsg") or "NEW"
            return OrderResponse(
                success=True,
                status=status,
                exchange_order_id=order_id,
                filled_quantity=order.quantity,
                avg_price=order.price,
                raw={"entry": raw},
            )
        except Exception as exc:  # pragma: no cover - externo
            logger.exception("Error enviando orden a Bybit: %s", exc)
            return OrderResponse(success=False, status="ERROR", error=str(exc))

    def cancel_order(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
        request: CancelRequest,
        *,
        dry_run: bool = False,
    ) -> CancelResponse:
        request.validate()
        if dry_run:
            return CancelResponse(success=True, raw={"dry_run": True})
        try:
            client = self._build_client(credential)
            raw = client.cancel_order(
                category="linear",
                symbol=request.symbol,
                orderId=request.exchange_order_id,
                orderLinkId=request.client_order_id,
            )
            return CancelResponse(success=True, raw={"resp": raw})
        except Exception as exc:  # pragma: no cover - externo
            logger.exception("Error cancelando orden en Bybit: %s", exc)
            return CancelResponse(success=False, error=str(exc), raw={"error": str(exc)})

    def fetch_account_balance(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
    ) -> Dict[str, float]:
        try:
            client = self._build_client(credential)
            raw = client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            bal = raw.get("result", {}).get("list", [{}])[0].get("coin", [{}])[0].get("walletBalance")
            return {"USDT": float(bal) if bal is not None else 0.0}
        except Exception:  # pragma: no cover - externo
            return {"USDT": 0.0}
        return {"USDT": 0.0}


ExchangeRegistry.register(BybitClient)
```

### `trading/exchanges/dydx.py`

```python
from __future__ import annotations

"""
Cliente dYdX v4 (NodeClient).
- Usa endpoint gRPC seguro público (mainnet).
- Soporta múltiples usuarios (AccountManager).
- Envía órdenes LIMIT/POST-ONLY en quantums/subticks según clob_pair.
"""

import os
import asyncio
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict

import grpc
import requests
from dydx_v4_client.node.client import NodeClient
from dydx_v4_client.node.message import order_id, order as node_order
from dydx_v4_client.node.builder import TxOptions, Builder
from dydx_v4_client.wallet import Wallet
from dydx_v4_client.key_pair import KeyPair
from v4_proto.dydxprotocol.clob.order_pb2 import Order

from .base import ExchangeClient, ExchangeRegistry
from ..accounts.models import AccountConfig, ExchangeCredential, ExchangeEnvironment
from ..accounts.manager import AccountManager
from ..orders.models import CancelRequest, CancelResponse, OrderRequest, OrderResponse, OrderType
from ..utils.logging import get_logger

logger = get_logger("trading.exchanges.dydx")


DEFAULT_GRPC = os.getenv("DYDX_GRPC_HOST", "dydx-dao-grpc-1.polkachu.com:443")
MAINNET_CHAIN_ID = "dydx-mainnet-1"
MAINNET_CHAIN_DENOM = "adydx"
MAINNET_USDC_DENOM = "ibc/8E27BA2D5493AF5636760E354E46004562C46AB7EC0CC4C1CA14E9E20E2545B5"
INDEXER_URL = os.getenv("DYDX_INDEXER_URL", "https://indexer.dydx.trade/v4/perpetualMarkets")
INDEXER_BASE = os.getenv("DYDX_INDEXER_BASE", "https://indexer.dydx.trade/v4")


def get_dydx_position(wallet_address: str, market_symbol: str, subaccount_number: int = 0) -> float:
    """
    Consulta la posición actual en dYdX usando el indexer REST API.
    
    Args:
        wallet_address: Dirección de la wallet (dydx1...)
        market_symbol: Símbolo del mercado (ej: ETH-USD)
        subaccount_number: Número de subaccount (default: 0)
    
    Returns:
        Cantidad firmada de la posición (long >0, short <0, 0 si no hay posición)
    """
    try:
        # El indexer de dYdX v4 usa el endpoint de subaccounts
        url = f"{INDEXER_BASE}/addresses/{wallet_address}/subaccountNumber/{subaccount_number}/perpetualPositions"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        positions = data.get("positions") or []
        market_symbol_upper = market_symbol.upper()
        
        for pos in positions:
            if pos.get("market") == market_symbol_upper:
                size = float(pos.get("size", "0") or "0")
                # dYdX devuelve size como string, positivo para long, negativo para short
                return size
        
        return 0.0
    except Exception as exc:
        logger.error("Error consultando posición dYdX para %s en %s: %s", wallet_address, market_symbol, exc)
        return 0.0


def close_dydx_position_via_order_executor(
    account: AccountConfig,
    credential: ExchangeCredential,
    symbol: str,
    position_size: float,
) -> bool:
    """
    Cierra una posición en dYdX usando OrderExecutor (reduceOnly MARKET).
    
    Args:
        account: Configuración de la cuenta
        credential: Credenciales del exchange
        symbol: Símbolo del mercado
        position_size: Tamaño de la posición (positivo para long, negativo para short)
    
    Returns:
        True si la orden se envió correctamente, False en caso contrario
    """
    if position_size == 0:
        return False
    
    try:
        from ..orders.executor import OrderExecutor
        from ..orders.models import OrderRequest, OrderSide, OrderType, TimeInForce
        
        # Crear AccountManager con la cuenta y credencial
        manager = AccountManager([account])
        
        executor = OrderExecutor(manager)
        
        # Determinar side: si position_size > 0 (long), necesitamos vender (SELL)
        # Si position_size < 0 (short), necesitamos comprar (BUY)
        side = OrderSide.SELL if position_size > 0 else OrderSide.BUY
        qty = abs(position_size)
        
        order = OrderRequest(
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            quantity=qty,
            price=None,
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )
        
        response = executor.execute(account.user_id, credential.exchange, order, dry_run=False)
        return response.success
    except Exception as exc:
        logger.error("Error cerrando posición dYdX: %s", exc)
        return False


class DydxClientWrapper(ExchangeClient):
    name = "dydx"

    def _make_channel(self, host: str) -> grpc.Channel:
        opts = [
            ("grpc.keepalive_time_ms", 3000),
            ("grpc.keepalive_timeout_ms", 1000),
            ("grpc.keepalive_permit_without_calls", True),
        ]
        return grpc.secure_channel(host, grpc.ssl_channel_credentials(), options=opts)

    def _build_client(self, credential: ExchangeCredential, *, host: str | None = None) -> NodeClient:
        api_key, api_secret = credential.resolve_keys(os.environ)
        if not api_key or not api_secret:
            raise ValueError("Faltan API key/secret para dYdX")
        node_host = (host or DEFAULT_GRPC).replace("https://", "").replace("http://", "")
        channel = self._make_channel(node_host)
        builder = Builder(chain_id=MAINNET_CHAIN_ID, denomination=MAINNET_CHAIN_DENOM)
        return NodeClient(channel=channel, builder=builder)

    def _resolve_wallet(self, client: NodeClient, private_hex: str, address: str) -> Wallet:
        private_clean = private_hex[2:] if private_hex.startswith("0x") else private_hex
        kp = KeyPair.from_hex(private_clean)
        acct = asyncio.run(client.get_account(address))
        return Wallet(kp, acct.account_number, acct.sequence)

    def _resolve_clob_pair(self, market_symbol: str) -> dict:
        """
        Busca clob_pair_id y parámetros de tick/step vía indexer REST.
        """
        market_symbol = market_symbol.upper()
        try:
            r = requests.get(INDEXER_URL, timeout=10)
            r.raise_for_status()
            data = r.json()
            markets = data.get("markets") or data.get("perpetualMarkets") or {}
            info = markets.get(market_symbol)
            if not info:
                raise ValueError(f"Mercado {market_symbol} no encontrado en indexer")
            return {
                "id": int(info["clobPairId"]),
                "quantum_conversion_exponent": int(info.get("quantumConversionExponent", -9)),
                "subticks_per_tick": int(info.get("subticksPerTick", 100000)),
                "step_base_quantums": int(info.get("stepBaseQuantums", 1000000)),
            }
        except Exception as exc:
            raise ValueError(f"No se pudo resolver mercado {market_symbol} via indexer: {exc}")

    def _quantize_qty(self, qty: float, step: Decimal) -> int:
        dv = (Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_DOWN)
        if dv <= 0:
            dv = Decimal(1)
        return int(dv * step)

    def _quantize_price(self, price: float, subticks_per_tick: int) -> int:
        return int(Decimal(str(price)) * Decimal(subticks_per_tick))

    def place_order(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
        order: OrderRequest,
        *,
        dry_run: bool = False,
    ) -> OrderResponse:
        order.validate()
        if dry_run:
            return OrderResponse(
                success=True,
                status="SIMULATED",
                exchange_order_id=None,
                filled_quantity=order.quantity,
                avg_price=order.price,
                raw={"dry_run": True, "symbol": order.symbol, "side": order.side.value, "price": order.price},
            )

        # Campos requeridos en credencial: api_key_env=wallet address, api_secret_env=permissioned private key
        api_key, api_secret = credential.resolve_keys(os.environ)
        subaccount_number = credential.extra.get("subaccount", 0) if credential.extra else 0
        market_symbol = credential.extra.get("symbol", order.symbol) if credential.extra else order.symbol

        try:
            client = self._build_client(credential)
        except Exception as exc:
            logger.error("dYdX error construyendo cliente: %s", exc)
            return OrderResponse(success=False, status="ERROR", error=str(exc))

        try:
            wallet = self._resolve_wallet(client, api_secret, api_key)
            pair = self._resolve_clob_pair(market_symbol)

            # dYdX usa quantums/subticks. Simplificamos: qty en step_base_quantums, price en subticks_per_tick.
            step = Decimal(str(pair["step_base_quantums"]))
            qty_quantums = self._quantize_qty(order.quantity, step)
            subticks_per_tick = int(pair["subticks_per_tick"])
            
            # Para órdenes MARKET, obtener precio actual del mercado
            if order.type == OrderType.MARKET and (order.price is None or order.price == 0):
                # Obtener precio actual desde el indexer
                try:
                    # Usar el mismo endpoint que _resolve_clob_pair
                    r = requests.get(INDEXER_URL, timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    markets = data.get("markets") or data.get("perpetualMarkets") or {}
                    market_info = markets.get(market_symbol.upper())
                    if not market_info:
                        raise ValueError(f"Mercado {market_symbol.upper()} no encontrado")
                    current_price = float(market_info.get("indexPrice") or market_info.get("markPrice") or market_info.get("oraclePrice", 0))
                    if current_price == 0:
                        raise ValueError("No se pudo obtener precio actual")
                    # Usar precio actual con un pequeño slippage para asegurar ejecución
                    # Para BUY: precio ligeramente más alto, para SELL: precio ligeramente más bajo
                    slippage = 0.001  # 0.1% slippage
                    if order.side.value.upper() == "BUY":
                        market_price = current_price * (1 + slippage)
                    else:
                        market_price = current_price * (1 - slippage)
                    logger.info("Orden MARKET: usando precio de mercado %f (precio actual: %f)", market_price, current_price)
                except Exception as exc:
                    logger.error("Error obteniendo precio para orden MARKET: %s", exc)
                    return OrderResponse(success=False, status="ERROR", error=f"No se pudo obtener precio para orden MARKET: {exc}")
            else:
                market_price = order.price or 0
                if market_price == 0:
                    return OrderResponse(success=False, status="ERROR", error="Precio requerido para órdenes LIMIT")
            
            price_subticks = self._quantize_price(market_price, subticks_per_tick)

            side = Order.Side.SIDE_BUY if order.side.value.upper() == "BUY" else Order.Side.SIDE_SELL
            # Bloque de expiración corto: 50 bloques (~) a futuro
            current_block = asyncio.run(client.latest_block_height())
            good_til_block = int(current_block + 50)
            order_flags = 0  # sin flags especiales
            oid = order_id(api_key, subaccount_number, client_id=1, clob_pair_id=int(pair["id"]), order_flags=order_flags)
            msg_order = node_order(
                order_id=oid,
                side=side,
                quantums=qty_quantums,
                subticks=price_subticks,
                time_in_force=Order.TimeInForce.TIME_IN_FORCE_UNSPECIFIED,
                reduce_only=order.reduce_only,
                good_til_block=good_til_block,
            )
            tx_opts = TxOptions(
                authenticators=[],
                sequence=wallet.sequence,
                account_number=wallet.account_number,
            )
            resp = asyncio.run(client.place_order(wallet, msg_order, tx_options=tx_opts))
            return OrderResponse(
                success=True,
                status="NEW",
                exchange_order_id=None,
                filled_quantity=float(order.quantity),
                avg_price=order.price,
                raw={"resp": str(resp)},
            )
        except Exception as exc:
            logger.error("dYdX error enviando orden: %s", exc)
            return OrderResponse(success=False, status="ERROR", error=str(exc))

    def cancel_order(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
        request: CancelRequest,
        *,
        dry_run: bool = False,
    ) -> CancelResponse:
        request.validate()
        return CancelResponse(success=True, raw={"dry_run": True})

    def fetch_account_balance(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
    ) -> Dict[str, float]:
        return {"USDC": 0.0}

    def cancel_all(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
        *,
        symbol: str,
        dry_run: bool = False,
    ) -> CancelResponse:
        # Cancelar en lote requiere armar estructuras de batch; por ahora devolvemos dry-run/ack.
        return CancelResponse(success=True, raw={"info": "cancel_all no implementado; noop"})


ExchangeRegistry.register(DydxClientWrapper)
```

### `trading/orders/__init__.py`

```python
"""
Modelos y helpers relacionados con órdenes.
"""

from .models import (
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderType,
    TimeInForce,
)

__all__ = [
    "OrderRequest",
    "OrderResponse",
    "OrderSide",
    "OrderType",
    "TimeInForce",
]
```

### `trading/orders/models.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good 'til cancelled
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill
    GTE_GTC = "GTE_GTC"  # Good till expiry (futuros Binance)


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    client_order_id: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.symbol:
            raise ValueError("OrderRequest.symbol vacío.")
        if self.quantity <= 0:
            raise ValueError("OrderRequest.quantity debe ser mayor a cero.")
        if self.type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and (self.price is None or self.price <= 0):
            raise ValueError("Las órdenes LIMIT requieren price > 0.")


@dataclass(slots=True)
class OrderResponse:
    success: bool
    status: str
    exchange_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    avg_price: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.success and not self.error:
            self.error = "Unknown error"


@dataclass(slots=True)
class CancelRequest:
    symbol: str
    exchange_order_id: Optional[str] = None
    client_order_id: Optional[str] = None

    def validate(self) -> None:
        if not self.symbol:
            raise ValueError("CancelRequest.symbol vacío.")
        if not (self.exchange_order_id or self.client_order_id):
            raise ValueError("CancelRequest requiere exchange_order_id o client_order_id.")


@dataclass(slots=True)
class CancelResponse:
    success: bool
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
```

### `trading/orders/executor.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

from ..accounts.manager import AccountManager
from ..orders.models import OrderRequest, OrderResponse
from ..exchanges.base import ExchangeClient, ExchangeRegistry
from ..accounts.models import AccountConfig, ExchangeCredential


@dataclass(slots=True)
class ExecutionContext:
    account: AccountConfig
    credential: ExchangeCredential
    exchange_client: ExchangeClient


class OrderExecutor:
    """
    Orquestador que toma una señal genérica y la envía al exchange correspondiente.
    """

    def __init__(self, account_manager: AccountManager):
        self._accounts = account_manager

    def _resolve_context(self, user_id: str, exchange_name: str) -> ExecutionContext:
        account = self._accounts.get_account(user_id)
        credential = account.get_exchange(exchange_name)
        client_cls: Type[ExchangeClient] = ExchangeRegistry.get(exchange_name)
        client = client_cls()
        return ExecutionContext(account=account, credential=credential, exchange_client=client)

    def execute(self, user_id: str, exchange_name: str, order: OrderRequest, *, dry_run: bool = True) -> OrderResponse:
        order.validate()
        ctx = self._resolve_context(user_id, exchange_name)
        return ctx.exchange_client.place_order(ctx.account, ctx.credential, order, dry_run=dry_run)
```

### `watcher_alertas.py`

```python
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
TRADING_DEFAULT_QTY = os.getenv("WATCHER_TRADING_DEFAULT_QTY", "0.01")
# Si se indica, calcula cantidad a partir de un notional USDT (qty = notional / price)
TRADING_DEFAULT_NOTIONAL_USDT = float(os.getenv("WATCHER_TRADING_NOTIONAL_USDT", "0") or 0)
TRADING_DRY_RUN = os.getenv("WATCHER_TRADING_DRY_RUN", "true").lower() != "false"
TRADING_MIN_PRICE = float(os.getenv("WATCHER_TRADING_MIN_PRICE", "0"))
TRADING_MIN_NOTIONAL = float(os.getenv("WATCHER_MIN_NOTIONAL_USDT", "20"))

_executor: OrderExecutor | None = None
_account_manager: AccountManager | None = None
_last_order_direction: dict[tuple[str, str], str] = {}
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
    Solo implementado para binance; si falla devuelve 0.
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
            print(
                f"[WATCHER][TRADE] user={user_id} ex={exchange} success={response.success} status={response.status} raw={response.raw}"
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
```

### `alerts.py`

```python
import os
import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

from paginado_binance import fetch_klines_paginado, INTERVAL_MS
from tabla_alertas import log_stream_bar
from backtest.realtime_backtest import process_realtime_signal, evaluate_realtime_risk
from velas import (
    SYMBOL_DISPLAY,
    API_SYMBOL,
    STREAM_INTERVAL,
    BB_DIRECTION,
    BB_LENGTH,
    BB_MULT,
    compute_bollinger_bands,
)


ALERT_STREAM_BARS = int(os.getenv("ALERT_STREAM_BARS", "5000"))
STOP_LOSS_PCT = float(os.getenv("STRAT_STOP_LOSS_PCT", "0.05"))
TAKE_PROFIT_PCT = float(os.getenv("STRAT_TAKE_PROFIT_PCT", "0.095"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_chat_ids_raw = os.getenv("TELEGRAM_CHAT_IDS", "")
TELEGRAM_CHAT_IDS = [part.strip() for part in _chat_ids_raw.replace(";", ",").split(",") if part.strip()]
SIGNAL_ALERTS_ENABLED = os.getenv("ALERT_ENABLE_BOLLINGER_SIGNALS", "false").lower() == "true"
_last_direction: str | None = None
_pending_break: dict | None = None  # Guarda rotura pendiente hasta que haya cierre de rebote

LOCAL_TZ_NAME = os.getenv("TZ", "UTC")
try:
    LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
except Exception:
    LOCAL_TZ = ZoneInfo("UTC")


def _prepare_frames() -> dict | None:
    df_stream = fetch_klines_paginado(API_SYMBOL, STREAM_INTERVAL, ALERT_STREAM_BARS)
    if df_stream.empty:
        return None

    ohlc_stream = df_stream[["Open", "High", "Low", "Close", "Volume"]].copy()
    if "CloseTimeDT" in df_stream.columns:
        ohlc_stream["BarCloseTime"] = df_stream["CloseTimeDT"]
    else:
        interval_ms = INTERVAL_MS.get(STREAM_INTERVAL, 0)
        ohlc_stream["BarCloseTime"] = df_stream.index + pd.to_timedelta(interval_ms, unit="ms")
    bb = compute_bollinger_bands(ohlc_stream, BB_LENGTH, BB_MULT)
    bb_aligned = bb.reindex(ohlc_stream.index).ffill()

    return {
        "stream": ohlc_stream,
        "bollinger": bb_aligned,
    }


def _bollinger_alert(bb_aligned: pd.DataFrame, ohlc_stream: pd.DataFrame):
    if bb_aligned is None or bb_aligned.empty or ohlc_stream.empty:
        return None

    close_series = ohlc_stream["Close"].astype("float64")
    upper = bb_aligned.get("upper")
    lower = bb_aligned.get("lower")
    basis = bb_aligned.get("basis")

    if upper is None or lower is None or close_series.empty:
        return None

    # Usa solo velas cerradas: descarta la última fila (vela en curso)
    if len(close_series) < 2 or len(upper) < 2 or len(lower) < 2:
        return None
    closed_close = close_series.iloc[:-1]
    closed_upper = upper.iloc[:-1]
    closed_lower = lower.iloc[:-1]
    closed_basis = basis.iloc[:-1] if basis is not None else None

    if closed_close.empty:
        return None

    last_idx = closed_close.index[-1]
    close_now = float(closed_close.iloc[-1])
    upper_now = float(closed_upper.iloc[-1])
    lower_now = float(closed_lower.iloc[-1])

    if any(np.isnan(val) for val in (close_now, upper_now, lower_now)):
        return None

    direction_filter = BB_DIRECTION

    global _pending_break
    trend = None
    direction = None
    ref_price = None
    trigger_price = None
    band_ref = None
    break_ts = None

    # Si hay una rotura pendiente, esperar rebote (cierre del lado opuesto de la banda) en vela posterior
    if _pending_break:
        pend_dir = _pending_break.get("direction")
        break_ts = _pending_break.get("break_ts")
        if pend_dir == "long" and direction_filter != -1:
            if break_ts is not None and last_idx > break_ts and close_now > lower_now:
                trend = "alcista"
                direction = "long"
                # Entrada a precio de cierre de la vela de rebote; banda se mantiene como referencia
                ref_price = close_now
                trigger_price = close_now
                band_ref = lower_now
                print(
                    f"[ALERT][PENDING] Consumida rotura pendiente LONG (rebote) band={lower_now:.2f} close={close_now:.2f} ts={last_idx}"
                )
                _pending_break = None
        elif pend_dir == "short" and direction_filter != 1:
            if break_ts is not None and last_idx > break_ts and close_now < upper_now:
                trend = "bajista"
                direction = "short"
                ref_price = close_now
                trigger_price = close_now
                band_ref = upper_now
                print(
                    f"[ALERT][PENDING] Consumida rotura pendiente SHORT (rebote) band={upper_now:.2f} close={close_now:.2f} ts={last_idx}"
                )
                _pending_break = None
        # Si no se cumplió el rebote, seguimos esperando (no devolvemos alerta aún)

    # Si no hay alerta confirmada, registrar nuevas roturas
    if trend is None:
        # Rotura long: cierre por debajo de la banda inferior
        if close_now < lower_now and direction_filter != -1:
            if _pending_break and _pending_break.get("direction") != "long":
                print(
                    f"[ALERT][PENDING] Reset por cambio de tendencia (prev={_pending_break}); nueva LONG ts={last_idx} band={lower_now:.2f} close={close_now:.2f}"
                )
            print(
                f"[ALERT][PENDING] Set LONG ts={last_idx} band={lower_now:.2f} close={close_now:.2f} upper={upper_now:.2f} lower={lower_now:.2f}"
            )
            _pending_break = {"direction": "long", "band": lower_now, "break_ts": last_idx}
        # Rotura short: cierre por encima de la banda superior
        elif close_now > upper_now and direction_filter != 1:
            if _pending_break and _pending_break.get("direction") != "short":
                print(
                    f"[ALERT][PENDING] Reset por cambio de tendencia (prev={_pending_break}); nueva SHORT ts={last_idx} band={upper_now:.2f} close={close_now:.2f}"
                )
            print(
                f"[ALERT][PENDING] Set SHORT ts={last_idx} band={upper_now:.2f} close={close_now:.2f} upper={upper_now:.2f} lower={lower_now:.2f}"
            )
            _pending_break = {"direction": "short", "band": upper_now, "break_ts": last_idx}
        return None

    # Si llegamos aquí es porque se confirmó un rebote y se va a emitir señal

    stop_loss = None
    take_profit = None
    if ref_price and ref_price > 0:
        if STOP_LOSS_PCT > 0:
            stop_loss = ref_price * (1 - STOP_LOSS_PCT) if direction == "long" else ref_price * (1 + STOP_LOSS_PCT)
        if TAKE_PROFIT_PCT > 0:
            take_profit = ref_price * (1 + TAKE_PROFIT_PCT) if direction == "long" else ref_price * (1 - TAKE_PROFIT_PCT)

    last_bar = ohlc_stream.iloc[-1]
    bar_close_ts = last_bar.get("BarCloseTime", last_idx)
    volume = float(last_bar.get("Volume", np.nan))
    basis_now = float(basis.iloc[-1]) if basis is not None else np.nan

    timestamp = bar_close_ts if isinstance(bar_close_ts, pd.Timestamp) else pd.Timestamp(bar_close_ts)

    return {
        "type": "bollinger_signal",
        # Timestamp exacto de cierre de la vela de rebote (sin desplazar al inicio)
        "timestamp": timestamp,
        "message": (
            f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: Señal Bollinger {trend} en {trigger_price:.2f} "
            f"(banda de ruptura {band_ref:.2f} ref)"
        ),
        # Entrada al cierre de la vela de rebote (market); se deja banda como referencia
        "price": trigger_price,
        "entry_price": ref_price,
        "close_price": close_now,
        "direction": direction,
        "basis": basis_now,
        "reference_band": band_ref,
        "volume": volume,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        # Alias para TP/SL consumidos por watcher_alertas
        "sl": stop_loss,
        "tp": take_profit,
    }



def generate_alerts() -> list[dict]:
    frames = _prepare_frames()
    if not frames:
        return []

    log_stream_bar(frames["stream"])
    try:
        evaluate_realtime_risk(frames["stream"], profile="tr")
    except Exception as exc:
        print(f"[ALERT][WARN] No se pudo evaluar SL/TP en tiempo real ({exc})")
    if not SIGNAL_ALERTS_ENABLED:
        return []

    alert = _bollinger_alert(frames["bollinger"], frames["stream"])
    if alert:
        try:
            process_realtime_signal(alert, profile="tr")
        except Exception as exc:
            print(f"[ALERT][WARN] No se pudo actualizar el backtest en tiempo real ({exc})")
        return [alert]
    return []


def format_alert_message(alert: dict) -> str:
    ts = alert.get("timestamp")
    ts_str = ""

    if isinstance(ts, pd.Timestamp):
        try:
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts_local = ts.tz_convert(LOCAL_TZ)
            ts_str = ts_local.isoformat()
        except Exception:
            ts_str = str(ts)
    elif hasattr(ts, "astimezone"):
        try:
            ts_local = ts.astimezone(LOCAL_TZ)
            ts_str = ts_local.isoformat()
        except Exception:
            ts_str = str(ts)
    elif hasattr(ts, "isoformat"):
        ts_str = ts.isoformat()
    else:
        ts_str = str(ts)

    base = f"{ts_str}\n{alert.get('message', '')}"

    sl = alert.get("stop_loss")
    tp = alert.get("take_profit")
    parts = []
    try:
        if tp is not None:
            parts.append(f"TP: {float(tp):.2f}")
        if sl is not None:
            parts.append(f"SL: {float(sl):.2f}")
    except Exception:
        pass

    if parts:
        base = f"{base}\n" + " | ".join(parts)

    return base


def send_alerts(alerts: list[dict]) -> int:
    if not alerts:
        return 0
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        return 0

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    sent = 0

    for alert in alerts:
        text = format_alert_message(alert)
        for chat_id in TELEGRAM_CHAT_IDS:
            payload = {
                "chat_id": chat_id,
                "text": text,
            }
            try:
                resp = requests.post(base_url, json=payload, timeout=10)
                resp.raise_for_status()
                sent += 1
            except Exception as exc:
                details = ""
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    try:
                        details = f" | Response: {exc.response.json()}"
                    except ValueError:
                        details = f" | Response: {exc.response.text}"
                print(f"[ERROR] Telegram send failed ({chat_id}): {exc}{details}")

    return sent


if __name__ == "__main__":
    alerts = generate_alerts()
    for alert in alerts:
        print(f"[ALERTA] {format_alert_message(alert)}")

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS:
        sent = send_alerts(alerts)
        print(f"[INFO] Alertas enviadas a Telegram: {sent}")
    else:
        print("[WARN] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_IDS no configurados; no se enviaron mensajes.")
```

### `velas.py`

```python
import os
import pandas as pd

from dotenv import load_dotenv

from paginado_binance import fetch_klines_paginado


load_dotenv()

SYMBOL_DISPLAY = os.getenv("SYMBOL", "ETHUSDT.P")
API_SYMBOL = SYMBOL_DISPLAY.replace(".P", "")

STREAM_INTERVAL = os.getenv("STREAM_INTERVAL", "30m").strip()
BB_LENGTH = int(os.getenv("BB_LENGTH", "20"))
BB_MULT = float(os.getenv("BB_MULT", "2.0"))
BB_DIRECTION = int(os.getenv("BB_DIRECTION", "0"))
try:
    BB_STD_DDOF = int(os.getenv("BB_STD_DDOF", "1"))
except ValueError:
    BB_STD_DDOF = 1
BB_STD_DDOF = max(BB_STD_DDOF, 0)


def fetch_stream_ohlc(limit: int) -> pd.DataFrame:
    df = fetch_klines_paginado(API_SYMBOL, STREAM_INTERVAL, limit)
    if df.empty:
        return df
    return df[["Open", "High", "Low", "Close", "Volume"]]


def compute_bollinger_bands(df: pd.DataFrame, length: int, mult: float) -> pd.DataFrame:
    if df is None or df.empty:
        idx = df.index if df is not None else None
        return pd.DataFrame(index=idx)

    length = max(int(length), 1)
    mult = float(mult)

    close = df["Close"].astype("float64")
    basis = close.rolling(length, min_periods=1).mean()
    deviation = close.rolling(length, min_periods=1).std(ddof=BB_STD_DDOF)
    upper = basis + mult * deviation
    lower = basis - mult * deviation

    idx = df.index
    return pd.DataFrame(
        {
            "basis": basis,
            "upper": upper,
            "lower": lower,
            "deviation": deviation,
            "close": close,
        },
        index=idx,
    )


def main():
    df = fetch_stream_ohlc(5000)
    bb = compute_bollinger_bands(df, BB_LENGTH, BB_MULT)
    if df.empty or bb.empty:
        print("[WARN] No se pudieron obtener datos.")
        return
    last = bb.iloc[-1]
    print("Última vela:", bb.index[-1])
    print("Bollinger Basis:", last.get("basis"))
    print("Bollinger Upper:", last.get("upper"))
    print("Bollinger Lower:", last.get("lower"))


if __name__ == "__main__":
    main()
```

### `telegram_bot_commands.py`

```python
# telegram_bot_commands.py
"""
Bot sencillo que atiende comandos de Telegram relacionados con la estrategia.
Actualmente soporta:
    /estavivo  -> devuelve el mismo estado que produce el heartbeat.
"""
from __future__ import annotations

import os
import time
from typing import Iterable, Optional

import requests
from dotenv import load_dotenv

from heartbeat_monitor import generate_heartbeat_message, required_processes_from_env


def _parse_chat_ids(chat_ids_env: str | None) -> list[str]:
    if not chat_ids_env:
        return []
    parts = [part.strip() for part in chat_ids_env.replace(";", ",").split(",")]
    return [part for part in parts if part]


def _send_message(token: str, chat_id: int | str, text: str, reply_to: Optional[int] = None) -> None:
    payload = {
        "chat_id": chat_id,
        "text": text,
    }
    if reply_to is not None:
        payload["reply_to_message_id"] = reply_to

    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
            timeout=10,
        ).raise_for_status()
    except Exception as exc:
        print(f"[BOT][WARN] No se pudo enviar respuesta a Telegram ({chat_id}): {exc}")


def _fetch_updates(token: str, offset: Optional[int]) -> dict:
    params = {
        "timeout": 30,
    }
    if offset is not None:
        params["offset"] = offset
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params=params,
            timeout=35,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[BOT][WARN] Error consultando getUpdates: {exc}")
        return {"ok": False, "result": []}


def _is_authorized(chat_id: int | str, allowed: Iterable[str]) -> bool:
    if not allowed:
        return True
    return str(chat_id) in allowed


def _normalize_command(text: str) -> str:
    if not text:
        return ""
    return text.strip().lower()


def _handle_command(
    *,
    token: str,
    chat_id: int,
    message_id: Optional[int],
    command: str,
    required_processes: list[str],
) -> None:
    if command.startswith("/estavivo"):
        report = generate_heartbeat_message(required_processes)
        _send_message(token, chat_id, report, reply_to=message_id)
        return

    if command in {"/start", "/help"}:
        help_text = (
            "Comandos disponibles:\n"
            "• /estavivo — chequea los procesos críticos y devuelve el estado actual.\n"
            "Los mensajes siguen el formato del heartbeat automático."
        )
        _send_message(token, chat_id, help_text, reply_to=message_id)
        return


def main() -> None:
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN no configurado.")

    allowed_chat_ids = _parse_chat_ids(os.getenv("TELEGRAM_CHAT_IDS"))
    required_processes = required_processes_from_env(None)
    if not required_processes:
        raise SystemExit("HEARTBEAT_PROCESSES vacío; definí procesos a monitorear.")

    print("[BOT] Telegram command listener iniciado.")
    offset: Optional[int] = None

    while True:
        data = _fetch_updates(token, offset)
        if not data.get("ok"):
            time.sleep(5)
            continue

        for update in data.get("result", []):
            offset = update["update_id"] + 1

            message = update.get("message") or update.get("channel_post")
            if not message:
                continue
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            if chat_id is None:
                continue
            if not _is_authorized(chat_id, allowed_chat_ids):
                continue

            text = message.get("text")
            command = _normalize_command(text)
            if not command or not command.startswith("/"):
                continue

            _handle_command(
                token=token,
                chat_id=chat_id,
                message_id=message.get("message_id"),
                command=command,
                required_processes=required_processes,
            )

        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[BOT] Finalizado por el usuario.")
```

### `check_telegram.py`

```python
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv(".env")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN vacío; configurá .env y reintentá.")

resp = requests.get(f"https://api.telegram.org/bot{TOKEN}/getMe", timeout=10)
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
```

### `trade_logger.py`

```python
# trade_logger.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo
from typing import Any
from datetime import datetime, timezone


DEFAULT_TRADES_PATH = os.getenv("STRAT_TRADES_CSV_PATH", "estrategia_trades.csv").strip()
SYMBOL_DISPLAY = os.getenv("SYMBOL", "ETHUSDT.P")
STREAM_INTERVAL = os.getenv("STREAM_INTERVAL", "30m").strip()
TZ_NAME = os.getenv("TZ", "UTC")
try:
    LOCAL_TZ = ZoneInfo(TZ_NAME)
except Exception:
    LOCAL_TZ = ZoneInfo("UTC")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_chat_ids_raw = os.getenv("TELEGRAM_CHAT_IDS", "")
TELEGRAM_CHAT_IDS = [part.strip() for part in _chat_ids_raw.replace(";", ",").split(",") if part.strip()]
TRADE_ALERTS_ENABLED = os.getenv("TRADE_ALERTS_ENABLED", "true").lower() == "true"

TRADE_COLUMNS = [
    "EntryTime",
    "OrderTime",
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

TRADE_TABLE_COLUMNS = [
    "EntryTime",
    "Direction",
    "ReferencePrice",
    "Fees",
    "PnLPct",
    "PnLAbs",
    "Source",
]

TRADE_LOG_SOURCE = (os.getenv("TRADE_LOG_SOURCE", "live") or "live").strip()
if not TRADE_LOG_SOURCE:
    TRADE_LOG_SOURCE = "live"
base_dashboard_dir = os.getenv("TRADES_DASHBOARD_BASE", "trades_dashboard").strip() or "trades_dashboard"
DEFAULT_TRADES_DASHBOARD_DIR = Path(base_dashboard_dir) / TRADE_LOG_SOURCE
TRADE_TABLE_CSV_PATH = Path(os.getenv("TRADE_TABLE_CSV_PATH", DEFAULT_TRADES_DASHBOARD_DIR / "trades_table.csv"))
TRADE_DASHBOARD_HTML_PATH = Path(os.getenv("TRADE_DASHBOARD_HTML_PATH", DEFAULT_TRADES_DASHBOARD_DIR / "trades_dashboard.html"))


def format_timestamp(ts: Any) -> str:
    try:
        if isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
        elif isinstance(ts, datetime):
            dt = ts
        elif hasattr(ts, "to_pydatetime"):
            dt = ts.to_pydatetime()
        else:
            dt = pd.Timestamp(ts).to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_local = dt.astimezone(LOCAL_TZ)
        return dt_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ts)


def _prepare_csv(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(path, index=False, encoding="utf-8")
        return

    try:
        existing = pd.read_csv(path)
    except Exception:
        return

    if any(col not in existing.columns for col in TRADE_COLUMNS):
        upgraded = existing.reindex(columns=TRADE_COLUMNS, fill_value=np.nan)
        upgraded.to_csv(path, index=False, encoding="utf-8")


def _ensure_trade_table():
    TRADE_TABLE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not TRADE_TABLE_CSV_PATH.exists():
        pd.DataFrame(columns=TRADE_TABLE_COLUMNS).to_csv(TRADE_TABLE_CSV_PATH, index=False, encoding="utf-8")


def _append_trade_table(entry_time: str, direction: str, entry_price: float, fees: float, pnl_abs: float, pnl_pct: float, *, source: str):
    try:
        _ensure_trade_table()
        pd.DataFrame(
            [{
                "EntryTime": entry_time,
                "Direction": direction,
                "ReferencePrice": entry_price,
                "Fees": fees,
                "PnLPct": pnl_pct,
                "PnLAbs": pnl_abs,
                "Source": source,
            }]
        ).to_csv(TRADE_TABLE_CSV_PATH, mode="a", header=False, index=False, encoding="utf-8")
    except Exception as exc:
        print(f"[TRADE][WARN] No se pudo actualizar trades_table.csv ({exc})")


def _render_trade_dashboard():
    try:
        if not TRADE_TABLE_CSV_PATH.exists():
            TRADE_DASHBOARD_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
            TRADE_DASHBOARD_HTML_PATH.write_text("<html><body><h2>Sin operaciones registradas</h2></body></html>", encoding="utf-8")
            return
        df = pd.read_csv(TRADE_TABLE_CSV_PATH)
        total = len(df)
        wins = int((df["PnLAbs"] > 0).sum())
        losses = int((df["PnLAbs"] < 0).sum())
        win_rate = (wins / total * 100) if total else 0.0
        avg_pct = df["PnLPct"].mean() * 100 if total else 0.0
        pnl_total = df["PnLAbs"].sum()
        summary_html = (
            "<table>"
            f"<tr><th>Total trades</th><td>{total}</td></tr>"
            f"<tr><th>Ganadores</th><td>{wins}</td></tr>"
            f"<tr><th>Perdedores</th><td>{losses}</td></tr>"
            f"<tr><th>Win rate</th><td>{win_rate:.2f}%</td></tr>"
            f"<tr><th>PnL promedio</th><td>{avg_pct:.2f}%</td></tr>"
            f"<tr><th>PnL total</th><td>{pnl_total:.2f}</td></tr>"
            "</table>"
        )
        table_html = df.to_html(index=False, classes="trade-table", float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x)
        html = f"""
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Trade Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; background: #111; color: #f5f5f5; }}
                h1 {{ color: #facc15; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
                th, td {{ border: 1px solid #333; padding: 0.5rem; text-align: left; }}
                th {{ background: #222; }}
                tr:nth-child(even) {{ background: #1a1a1a; }}
            </style>
        </head>
        <body>
            <h1>Registro de trades</h1>
            {summary_html}
            {table_html}
        </body>
        </html>
        """
        TRADE_DASHBOARD_HTML_PATH.write_text(html, encoding="utf-8")
    except Exception as exc:
        print(f"[TRADE][WARN] No se pudo generar dashboard de trades ({exc})")


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
    order_time: Any | None = None,
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
    outcome_label = "GANANCIA" if outcome == "win" else ("PÉRDIDA" if outcome == "loss" else "RESULTADO NEUTRO")

    order_time_obj = order_time or entry_time
    if isinstance(order_time_obj, pd.Timestamp):
        order_time_str = order_time_obj.isoformat()
    elif hasattr(order_time_obj, "isoformat"):
        order_time_str = order_time_obj.isoformat()
    elif order_time_obj is None:
        order_time_str = ""
    else:
        order_time_str = str(order_time_obj)

    data = {
        "EntryTime": entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time),
        "OrderTime": order_time_str,
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
        f"[TRADE] {SYMBOL_DISPLAY} {STREAM_INTERVAL} | {direction.upper()} {entry_reason} → {exit_reason} | "
        f"Entry {entry_price:.2f} Exit {exit_price:.2f} | Fees {fees:.2f} | PnL {pnl_abs:.2f} ({pnl_pct*100:.2f}%)"
    )
    try:
        _append_trade_table(
            entry_time=data["EntryTime"],
            direction=direction,
            entry_price=entry_price,
            fees=fees,
            pnl_abs=pnl_abs,
            pnl_pct=pnl_pct,
            source=TRADE_LOG_SOURCE,
        )
        _render_trade_dashboard()
    except Exception as exc:
        print(f"[TRADE][WARN] No se pudo actualizar el tablero de trades ({exc})")
    print(message)
    if notify:
        try:
            ts_entry = format_timestamp(entry_time)
            ts_exit = format_timestamp(exit_time)
            tele_msg = (
                f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}\n"
                f"Cierre {direction.upper()}\n"
                f"Entrada: {entry_price:.2f} ({ts_entry})\n"
                f"Salida: {exit_price:.2f} ({ts_exit})\n"
                f"Fees: {fees:.2f}\n"
                f"Resultado: {outcome_label} {pnl_abs:.2f} ({pnl_pct*100:+.2f}%)"
            )
            _send_trade_notification(tele_msg)
        except Exception as exc:
            print(f"[TRADE][WARN] Error enviando alerta de trade: {exc}")


def send_trade_notification(text: str):
    _send_trade_notification(text)
```

### `heartbeat_monitor.py`

```python
# heartbeat_monitor.py
"""
Heartbeat que envía cada cierto intervalo el estado de los procesos críticos
al bot de Telegram configurado en el .env.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from trade_logger import send_trade_notification
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from trade_logger import send_trade_notification  # type: ignore


DEFAULT_PROCESS_LIST = (
    "python watcher_alertas.py;"
    "python backtest/order_fill_listener.py;"
    "python estrategiaBollinger.py"
)


@dataclass
class ProcessStatus:
    label: str
    running: bool
    matches: list[str]


def _parse_required_processes(value: str | None) -> list[str]:
    if not value:
        value = DEFAULT_PROCESS_LIST
    parts = [part.strip() for part in value.replace(",", ";").split(";")]
    return [part for part in parts if part]


def required_processes_from_env(override: str | None = None) -> list[str]:
    """
    Devuelve la lista de procesos a monitorear usando la env HEARTBEAT_PROCESSES.
    """
    env_value = override if override is not None else os.getenv("HEARTBEAT_PROCESSES")
    return _parse_required_processes(env_value)


def _list_process_commands() -> Sequence[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,command"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"No se pudo obtener la lista de procesos ({exc})") from exc
    lines = result.stdout.splitlines()
    if not lines:
        return []
    return lines[1:]  # descarta encabezado


def _evaluate_processes(required: Iterable[str], processes: Sequence[str]) -> list[ProcessStatus]:
    statuses: list[ProcessStatus] = []
    for label in required:
        matches = [proc for proc in processes if label in proc]
        statuses.append(ProcessStatus(label=label, running=bool(matches), matches=matches))
    return statuses


def _build_message(
    *,
    statuses: Sequence[ProcessStatus],
    tz: ZoneInfo,
) -> str:
    now = datetime.now(tz)
    header = f"[HEARTBEAT] {now.isoformat(timespec='seconds')}"
    lines = [header, ""]
    overall = "OK" if all(status.running for status in statuses) else "ALERTA"
    lines.append(f"Estado general: {overall}")
    lines.append("")
    for status in statuses:
        state = "OK" if status.running else "FALTA"
        lines.append(f"- {state} :: {status.label}")
        if status.running and status.matches:
            first = status.matches[0].strip()
            lines.append(f"    {first}")
    return "\n".join(lines)


def _resolve_timezone() -> ZoneInfo:
    tz_name = os.getenv("TZ", "UTC")
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def run_heartbeat(
    *,
    required_processes: list[str],
    interval_hours: float,
    once: bool,
) -> None:
    tz = _resolve_timezone()
    sleep_seconds = max(1.0, interval_hours * 3600.0)

    while True:
        message = generate_heartbeat_message(required_processes, tz=tz)
        send_trade_notification(message)
        if once:
            break
        time.sleep(sleep_seconds)


def generate_heartbeat_message(
    required_processes: list[str],
    tz: ZoneInfo | None = None,
) -> str:
    """
    Construye el mensaje resumido de estado para los procesos indicados.
    """
    tz = tz or _resolve_timezone()
    processes = _list_process_commands()
    statuses = _evaluate_processes(required_processes, processes)
    return _build_message(statuses=statuses, tz=tz)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Heartbeat que avisa por Telegram si los procesos clave están activos."
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "12")),
        help="Intervalo entre notificaciones (en horas).",
    )
    parser.add_argument(
        "--processes",
        type=str,
        default=os.getenv("HEARTBEAT_PROCESSES"),
        help="Lista de procesos a monitorear (separador ';' o ',').",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Enviar solo una notificación y salir (útil para pruebas manuales).",
    )

    args = parser.parse_args()

    required_processes = required_processes_from_env(args.processes)
    if not required_processes:
        raise SystemExit("No se encontraron procesos a monitorear (revisá HEARTBEAT_PROCESSES).")

    run_heartbeat(
        required_processes=required_processes,
        interval_hours=max(0.01, args.interval_hours),
        once=args.once,
    )


if __name__ == "__main__":
    main()
```

### `paginado_binance.py`

```python
# paginado_binance.py
# ---------------------------------------------------------
# Descarga histórico de klines con PAGINADO (hasta N velas).
# Devuelve un DataFrame con columnas:
#   ["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"]
# Index: DatetimeIndex en tu TZ (por env TZ).
#
# ENV opcionales:
#   BINANCE_UM_BASE_URL=https://fapi.binance.com
#   TZ=America/Argentina/Buenos_Aires
#   PAGINATE_PAGE_LIMIT=1500      # tope por request (máx 1500 en Binance)
#   PAGE_SLEEP_SEC=0.2            # pausa entre requests
# ---------------------------------------------------------

import os
import time
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Cliente Binance
try:
    from binance.um_futures import UMFutures
except Exception:
    print("ERROR: Falta 'binance-futures-connector'. Instalá:")
    print("  pip install binance-futures-connector")
    raise

def _get_client():
    base_url = os.getenv("BINANCE_UM_BASE_URL", "https://fapi.binance.com")
    return UMFutures(base_url=base_url)

# ms por intervalo (los más comunes)
INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000
}

def fetch_klines_paginado(
    symbol: str,
    interval: str,
    total_bars: int,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    page_limit: int | None = None,
    sleep_sec: float | None = None,
    tz_name: str | None = None,
) -> pd.DataFrame:
    """
    Baja hasta total_bars velas paginando contra Binance.
    Si no especificás start/end, descarga hacia atrás desde 'now'.

    Retorna DataFrame con columnas:
      Open, High, Low, Close, Volume, CloseTime, CloseTimeDT
    Index TZ-aware (según TZ).
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Interval no soportado para paginar: {interval}")

    page_limit = int(page_limit or os.getenv("PAGINATE_PAGE_LIMIT", 1500))
    page_limit = max(1, min(page_limit, 1500))  # Binance no da más de 1500
    sleep_sec = float(sleep_sec or os.getenv("PAGE_SLEEP_SEC", 0.2))
    tz_name = tz_name or os.getenv("TZ", "America/Argentina/Buenos_Aires")

    ms_per = INTERVAL_MS[interval]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ms = end_ms or now_ms
    start_ms = start_ms or (end_ms - total_bars * ms_per)

    client = _get_client()
    out = []
    fetched = 0
    curr_start = start_ms

    while fetched < total_bars:
        if curr_start is not None and end_ms is not None and curr_start >= end_ms:
            break
        batch_limit = min(page_limit, total_bars - fetched)
        data = client.klines(
            symbol=symbol,
            interval=interval,
            startTime=curr_start,
            endTime=end_ms,
            limit=batch_limit
        )
        if not data:
            break

        out.extend(data)
        fetched += len(data)

        last_close = int(data[-1][6])
        next_start = last_close + 1
        # Si no avanzamos, evitamos loop infinito
        if next_start <= curr_start:
            break
        if end_ms is not None and next_start >= end_ms:
            break
        curr_start = next_start

        time.sleep(sleep_sec)

    if not out:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"])

    # Ensamble
    rows = []
    # Nos quedamos con las ÚLTIMAS total_bars por si juntamos de más
    for k in out[-total_bars:]:
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
    df["DateUTC"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df["CloseTimeDT_UTC"] = pd.to_datetime(df["CloseTime"], unit="ms", utc=True)

    tz = ZoneInfo(tz_name)
    df = df.set_index(df["DateUTC"].dt.tz_convert(tz)).sort_index()
    df["CloseTimeDT"] = df["CloseTimeDT_UTC"].dt.tz_convert(tz)

    return df[["Open","High","Low","Close","Volume","CloseTime","CloseTimeDT"]]
```

### `tabla_alertas.py`

```python
# tabla_alertas.py
import os
from pathlib import Path

import pandas as pd

from zoneinfo import ZoneInfo

ALERTS_TABLE_CSV_PATH = os.getenv("ALERTS_TABLE_CSV_PATH", "alerts_stream.csv").strip()
ALERTS_TABLE_TZ = os.getenv("TZ", "UTC")

try:
    _LOCAL_TZ = ZoneInfo(ALERTS_TABLE_TZ)
except Exception:
    _LOCAL_TZ = ZoneInfo("UTC")

CSV_COLUMNS = ["Timestamp", "TimestampUTC", "Open", "High", "Low", "Close", "Volume"]

_last_logged = None


def _ensure_header(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(path, index=False, encoding="utf-8")


def log_stream_bar(df: pd.DataFrame):
    """
    Registra la última vela disponible en el CSV definido por ALERTS_TABLE_CSV_PATH.
    Evita duplicados por timestamp.
    """
    global _last_logged

    if df is None or df.empty:
        return

    ts = df.index[-1]
    if _last_logged is not None and ts == _last_logged:
        return

    row = df.iloc[-1]
    try:
        ts_local = ts.tz_convert(_LOCAL_TZ) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(_LOCAL_TZ)
    except Exception:
        ts_local = ts
    try:
        ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    except Exception:
        ts_utc = ts

    data = {
        "Timestamp": ts_local.isoformat() if hasattr(ts_local, "isoformat") else str(ts_local),
        "TimestampUTC": ts_utc.isoformat() if hasattr(ts_utc, "isoformat") else str(ts_utc),
        "Open": row["Open"],
        "High": row["High"],
        "Low": row["Low"],
        "Close": row["Close"],
        "Volume": row["Volume"],
    }

    path = Path(ALERTS_TABLE_CSV_PATH)
    _ensure_header(path)
    pd.DataFrame([data]).to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    _last_logged = ts
```

### `estrategiaBollinger.py`

```python
# estrategiaBollinger.py
"""
Estrategia basada en señales de Bandas de Bollinger:
  - Consume las señales de `alerts.generate_alerts()`.
  - Cruce al alza de la banda inferior => abre LONG (si no hay posición en esa dirección).
  - Cruce a la baja de la banda superior => abre SHORT.
  - Cada cambio de tendencia cierra la posición vigente antes de abrir la nueva.
  - Solo imprime eventos / recomendaciones (sin ejecución real).
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from alerts import generate_alerts, format_alert_message
from trade_logger import log_trade, send_trade_notification, format_timestamp


POLL_SECONDS = float(os.getenv("STRAT_POLL_SECONDS", os.getenv("ALERT_POLL_SECONDS", "5")))
FEE_RATE = float(os.getenv("STRAT_FEE_RATE", "0.0005"))
SYMBOL_DISPLAY = os.getenv("SYMBOL", "ETHUSDT.P")
STREAM_INTERVAL = os.getenv("STREAM_INTERVAL", "30m").strip()


@dataclass
class Position:
    direction: str   # "long" o "short"
    entry_price: float
    opened_at: pd.Timestamp
    entry_reason: str


class StrategyState:
    def __init__(self):
        self.current_position: Optional[Position] = None

    def close_current(self, exit_price: float, exit_time: pd.Timestamp, exit_reason: str):
        if self.current_position is None:
            return
        pos = self.current_position
        fees = (pos.entry_price + exit_price) * FEE_RATE
        log_trade(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.opened_at,
            exit_time=exit_time,
            entry_reason=pos.entry_reason,
            exit_reason=exit_reason,
            fees=fees,
            notify=True,
        )
        self.current_position = None

    def open_position(self, direction: str, price: float, ts: pd.Timestamp, reason: str):
        if price <= 0:
            print("[STRAT][WARN] Precio inválido para abrir posición")
            return
        self.current_position = Position(direction, price, ts, reason)
        print(f"[STRAT] Nueva {direction.upper()} @ {price:.2f} (motivo: {reason})")
        try:
            ts_fmt = format_timestamp(ts)
            send_trade_notification(
                f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}\n"
                f"Apertura {direction.upper()}\n"
                f"Precio: {price:.2f}\n"
                f"Hora: {ts_fmt}\n"
                f"Motivo: {reason}"
            )
        except Exception as exc:
            print(f"[STRAT][WARN] Error notificando apertura: {exc}")


def _extract_price_from_alert(alert: dict) -> Optional[float]:
    msg = alert.get("message", "")
    for token in msg.split():
        try:
            val = float(token.replace(",", "."))
            if val > 0:
                return val
        except Exception:
            continue
    return None


def _direction_for_alert(alert: dict) -> Optional[str]:
    explicit = alert.get("direction")
    if isinstance(explicit, str) and explicit.lower() in ("long", "short"):
        return explicit.lower()

    typ = alert.get("type")
    msg = alert.get("message", "").lower()
    if typ == "bollinger_signal":
        if "alcista" in msg:
            return "long"
        if "bajista" in msg:
            return "short"
    return None


def run_strategy():
    state = StrategyState()
    print("[STRAT] Estrategia Bollinger iniciada")

    while True:
        try:
            alerts = generate_alerts()
        except Exception as exc:
            print(f"[STRAT][ERROR] {exc}")
            time.sleep(POLL_SECONDS)
            continue

        if alerts:
            for alert in alerts:
                msg = format_alert_message(alert)
                print(f"[STRAT][ALERTA] {msg}")

            for alert in alerts:
                direction = _direction_for_alert(alert)
                if direction is None:
                    continue

                price = alert.get("price")
                if price is not None:
                    try:
                        price = float(price)
                    except Exception:
                        price = None
                if price is None:
                    price = _extract_price_from_alert(alert)

                ts = alert.get("timestamp")
                if not isinstance(ts, pd.Timestamp):
                    try:
                        ts = pd.Timestamp(ts)
                    except Exception:
                        ts = pd.Timestamp.utcnow()

                if state.current_position is not None:
                    if state.current_position.direction == direction:
                        print(f"[STRAT] Señal {alert['type']} coincide con posición actual {direction.upper()}, se ignora.")
                        continue
                    exit_price = price if price is not None and price > 0 else state.current_position.entry_price
                    state.close_current(exit_price, ts, alert["type"])

                if price is None or price <= 0:
                    price = float(os.getenv("STRAT_FALLBACK_PRICE", "0"))
                if price <= 0:
                    print("[STRAT][WARN] No se pudo determinar precio de entrada; se omite la operación.")
                    continue

                state.open_position(direction, price, ts, alert["type"])

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_strategy()
```

### `gBollinger.py`

```python
import os

import numpy as np
import pandas as pd
import mplfinance as mpf

from paginado_binance import fetch_klines_paginado
from velas import (
    SYMBOL_DISPLAY,
    API_SYMBOL,
    STREAM_INTERVAL,
    BB_LENGTH,
    BB_DIRECTION,
    BB_MULT,
    compute_bollinger_bands,
)


PLOT_STREAM_BARS = int(os.getenv("PLOT_STREAM_BARS", "5000"))
WARN_TOO_MUCH = int(os.getenv("WARN_TOO_MUCH", "5000"))
BB_LINE_WIDTH = float(os.getenv("BB_LINE_WIDTH", "2.0"))
BB_BASIS_COLOR = os.getenv("BB_BASIS_COLOR", "#facc15")
BB_UPPER_COLOR = os.getenv("BB_UPPER_COLOR", "#1dac70")
BB_LOWER_COLOR = os.getenv("BB_LOWER_COLOR", "#dc2626")
BB_FILL_ALPHA = float(os.getenv("BB_FILL_ALPHA", "0.12"))
BB_SIGNAL_MARKER_COLOR = os.getenv("BB_SIGNAL_MARKER_COLOR", "white")
BB_SIGNAL_MARKER_SIZE = float(os.getenv("BB_SIGNAL_MARKER_SIZE", "80"))


def _style_tv_dark():
    mc = mpf.make_marketcolors(
        up="lime",
        down="red",
        edge="inherit",
        wick="white",
        volume="in",
    )
    return mpf.make_mpf_style(
        marketcolors=mc,
        base_mpf_style="nightclouds",
        facecolor="black",
        edgecolor="black",
        gridcolor="#333333",
        gridstyle="--",
        rc={"axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"},
        y_on_right=False,
    )


def _has_data(s: pd.Series | None) -> bool:
    if s is None or len(s) == 0:
        return False
    try:
        arr = pd.to_numeric(s, errors="coerce").to_numpy()
        if arr.size == 0:
            return False
        return np.isfinite(arr).any()
    except Exception:
        return False


def _compute_signal_points(
    ohlc: pd.DataFrame, upper: pd.Series | None, lower: pd.Series | None
) -> pd.Series | None:
    if ohlc is None or ohlc.empty or not _has_data(upper) or not _has_data(lower):
        return None

    close = pd.to_numeric(ohlc["Close"], errors="coerce").astype("float64")
    upper_vals = pd.to_numeric(upper, errors="coerce").astype("float64")
    lower_vals = pd.to_numeric(lower, errors="coerce").astype("float64")

    close_prev = close.shift(1)
    upper_prev = upper_vals.shift(1)
    lower_prev = lower_vals.shift(1)

    crossed_lower = (close_prev < lower_prev) & (close > lower_vals)
    crossed_upper = (close_prev > upper_prev) & (close < upper_vals)

    signals = pd.Series(np.nan, index=close.index, dtype="float64")

    if BB_DIRECTION != -1:
        long_mask = crossed_lower.fillna(False)
        signals.loc[long_mask] = lower_vals.loc[long_mask]
    if BB_DIRECTION != 1:
        short_mask = crossed_upper.fillna(False)
        signals.loc[short_mask] = upper_vals.loc[short_mask]
    return signals if signals.notna().any() else None


def main():
    df_stream = fetch_klines_paginado(API_SYMBOL, STREAM_INTERVAL, PLOT_STREAM_BARS)
    if df_stream.empty:
        raise SystemExit("[ERROR] No se pudieron obtener velas del stream")

    ohlc_stream = df_stream[["Open", "High", "Low", "Close", "Volume"]]
    bb = compute_bollinger_bands(ohlc_stream, BB_LENGTH, BB_MULT).reindex(ohlc_stream.index).ffill()

    basis = bb.get("basis")
    upper = bb.get("upper")
    lower = bb.get("lower")

    addplots = []

    if _has_data(basis):
        addplots.append(mpf.make_addplot(basis, color=BB_BASIS_COLOR, width=BB_LINE_WIDTH, ylabel="Bollinger"))
    if _has_data(upper):
        addplots.append(mpf.make_addplot(upper, color=BB_UPPER_COLOR, width=BB_LINE_WIDTH, linestyle="--"))
    if _has_data(lower):
        addplots.append(mpf.make_addplot(lower, color=BB_LOWER_COLOR, width=BB_LINE_WIDTH, linestyle="--"))

    if _has_data(upper) and _has_data(lower):
        upper_vals = pd.to_numeric(upper, errors="coerce").to_numpy(dtype="float64")
        lower_vals = pd.to_numeric(lower, errors="coerce").to_numpy(dtype="float64")
        mask = np.isfinite(upper_vals) & np.isfinite(lower_vals)
        if mask.any():
            addplots.append(
                mpf.make_addplot(
                    upper,
                    color=BB_UPPER_COLOR,
                    width=0,
                    fill_between=dict(
                        y1=upper_vals,
                        y2=lower_vals,
                        where=mask,
                        alpha=BB_FILL_ALPHA,
                        color=BB_UPPER_COLOR,
                    ),
                )
            )

    signal_points = _compute_signal_points(ohlc_stream, upper, lower)
    if signal_points is not None:
        addplots.append(
            mpf.make_addplot(
                signal_points,
                type="scatter",
                marker="o",
                color=BB_SIGNAL_MARKER_COLOR,
                markersize=BB_SIGNAL_MARKER_SIZE,
            )
        )

    mpf.plot(
        ohlc_stream,
        type="candle",
        style=_style_tv_dark(),
        addplot=addplots,
        figsize=(12, 6),
        datetime_format="%Y-%m-%d %H:%M",
        title=f"{SYMBOL_DISPLAY} {STREAM_INTERVAL} — Bandas de Bollinger",
        warn_too_much_data=WARN_TOO_MUCH,
    )


if __name__ == "__main__":
    main()
```

### `backtest/README.md`

```markdown
# Backtests de la estrategia Bollinger

Este directorio reúne las utilidades necesarias para ejecutar backtests de la estrategia de Bandas de Bollinger contra datos de Binance USD‑M Futures. Los flujos cubren dos perfiles de salida:

- **TR**: backtest corto para trading intradía (carpeta `backtestTR`).
- **Histórico**: corridas extensas para análisis de largo plazo (carpeta `backtestHistorico`).

Ambos perfiles comparten el mismo motor (`run_backtest.py`) y las rutas/alineaciones de archivos se controlan desde `config.py` y variables de entorno.

## Requisitos previos

- Python 3.11+ recomendado (el proyecto usa pandas, numpy, plotly, etc.).
- Dependencias instaladas: `pip install -r requirements.txt`.
- Variables de entorno cargadas (por ejemplo `source .venv/bin/activate && set -a && source .env && set +a`) para que `velas.py`, `trade_logger.py` y los backtests reciban el símbolo, intervalos, credenciales y rutas de salida.
- Acceso HTTPS al endpoint de Binance (`BINANCE_UM_BASE_URL`, por defecto `https://fapi.binance.com`).

## Archivos clave

- `run_backtest.py`: motor que descarga velas, genera señales Bollinger, simula entradas/salidas y guarda resultados.
- `build_dashboard.py`: genera un dashboard HTML con métricas, gráfico y detalle de operaciones.
- `config.py`: define perfiles `tr` y `historico`, junto con las rutas de CSV/PNG/HTML (pueden sobrescribirse vía variables como `STRAT_BACKTEST_TRADES_PATH`).
- `backtestTR/` y `backtestHistorico/`: carpetas destino para cada perfil (CSV de trades, gráfico y dashboard).

## Ejecutar el backtest

1. **Activar entorno** (si aplica):
   ```bash
   source .venv/bin/activate
   set -a && source .env && set +a  # opcional pero recomendado
   ```

2. **Lanzar el backtest**:

   - Perfil TR (últimas semanas, salidas en `backtest/backtestTR/`):
     ```bash
     python backtest/run_backtest.py --profile tr --weeks 2
     ```
     Ajustá `--weeks` según la ventana que quieras analizar (si lo omitís, usa `BACKTEST_STREAM_BARS`).

   - Perfil Histórico (meses de datos, salidas en `backtest/backtestHistorico/`):
     ```bash
     python backtest/run_backtest.py --profile historico --months 6
     ```
     También podés fijar fechas exactas:
     ```bash
     python backtest/run_backtest.py --profile historico --start 2024-01-01T00:00:00Z --end 2024-06-30T23:59:59Z
     ```

   Durante la ejecución se imprime la comisión estimada en Binance, el rango temporal efectivo y un resumen con métricas (trades totales, win rate, PnL, drawdown, fees).

### Parámetros útiles

`run_backtest.py` acepta varias banderas para ajustar la corrida:

- `--stream-bars`: cantidad base de velas a descargar (default `BACKTEST_STREAM_BARS`).
- `--profile {tr,historico}`: selecciona el preset de rutas definido en `config.py`.
- `--trades-out / --plot-out`: rutas de salida personalizadas para CSV/PNG.
- `--weeks` o `--months`: rango relativo hacia atrás (usar solo uno).
- `--start / --end`: fechas ISO8601 (UTC) para rango absoluto.
- `--show`: abre la figura de Matplotlib al terminar si `matplotlib` está disponible.

> Nota: el motor usa las mismas Bandas de Bollinger configuradas para el watcher (`BB_LENGTH`, `BB_MULT`, `BB_DIRECTION`, `STREAM_INTERVAL`, etc.), por lo que cualquier cambio en `.env` impactará tanto las señales en vivo como el backtest.

## Visualización y dashboards

Una vez generado el CSV de trades podés construir el dashboard interactivo:

```bash
python backtest/build_dashboard.py --profile tr --price alerts_stream.csv
```

- `--profile` funciona igual que en el backtest; si no lo indicás usa el valor por defecto (`BACKTEST_PROFILE` o `tr`).
- `--trades` permite elegir un CSV alternativo (por ejemplo una corrida histórica guardada en otro directorio).
- `--price` es opcional, pero al pasar el CSV de precios (`alerts_stream.csv` u otro con columnas `Timestamp` y `Close`) se superpone la curva de precios con las entradas/salidas.
- `--html` define el destino del dashboard (default según preset).
- `--show` abre automáticamente el HTML en el navegador.

Los dashboards incluyen resumen estadístico, PnL acumulado, histograma de rendimiento y tablas con los últimos trades/operaciones. El archivo resultante se guarda en `backtest/backtestTR/dashboard.html` o `backtest/backtestHistorico/dashboard.html` según el perfil elegido.

### Listener para minuto exacto de fills

Si querés capturar el minuto exacto en que se ejecuta una orden pendiente, corré el listener dedicado (opera sobre el mismo `realtime_state.json` que usa el backtest en vivo):

```bash
python backtest/order_fill_listener.py --profile tr
```

- Monitorea las órdenes con `status=pending` y consulta velas de 1 minuto para detectar el primer cruce del precio objetivo.
- Actualiza el estado a `open` con el timestamp de esa vela (UTC) y mantiene la misma lógica de SL/TP definida por la estrategia.
- Parámetros opcionales:
  - `--poll-seconds` (default 15) ajusta la frecuencia de consulta.
  - `--tolerance` permite sumar una tolerancia absoluta al match del precio.
  - `--lookback-minutes` define la ventana de búsqueda al reconstruir la vela que ejecutó la orden.

Mantenelo corriendo junto al watcher de señales si necesitás una simulación intradía con precisión de minuto.

### Heartbeat / Verificador de vida

Para recibir un aviso cada 3 horas (o la frecuencia que definas) indicando si los procesos críticos siguen activos, podés usar el heartbeat incluido en la raíz del repo:

```bash
python heartbeat_monitor.py --interval-hours 3
```

- Por defecto chequea que estén corriendo `python watcher_alertas.py`, `python backtest/order_fill_listener.py` y `python estrategiaBollinger.py`. Podés ajustar la lista con la variable `HEARTBEAT_PROCESSES`, usando `;` o `,` como separador.
- El heartbeat reutiliza el mismo bot/configuración de Telegram (variables `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_IDS`). A cada intervalo envía un mensaje con el resumen de estado.
- Para pruebas puntuales, agregá `--once` y solo mandará una notificación.

También podés lanzar el listener de comandos para responder manualmente en cualquier momento con `/estavivo` y obtener el mismo estado bajo demanda:

```bash
python telegram_bot_commands.py
```

- Reconoce `/start`, `/help`, `/estavivo`, `/ultimaalerta`, `/ultimatrade` y `/resumen`. Los tres últimos permiten consultar rápidamente la última señal, el último trade y un resumen de métricas del CSV configurado.
- El comando responde únicamente a los chats listados en `TELEGRAM_CHAT_IDS` (si está vacío, acepta a todos).

## Trading en exchanges (estructura preliminar)

El paquete `trading/` incorpora la base para ejecutar órdenes reales o simuladas, contemplando múltiples exchanges y usuarios:

- `trading/orders/models.py`: definiciones comunes (`OrderRequest`, `OrderResponse`, enums `OrderSide/OrderType/TimeInForce`).
- `trading/accounts/models.py`: descripciones de cuentas (`AccountConfig`, `ExchangeCredential`, ambientes testnet/live).
- `trading/accounts/manager.py`: carga configuraciones desde YAML/JSON y resuelve credenciales leyendo variables de entorno.
- `trading/exchanges/base.py`: interfaz `ExchangeClient` + `ExchangeRegistry` para registrar implementaciones por exchange.
- `trading/exchanges/binance.py`: cliente Binance en modo dry-run/testnet (no envía órdenes reales todavía).
- `trading/orders/executor.py`: orquestador que toma una señal genérica y la envía al exchange adecuado.

### Archivo de cuentas

Ejemplo (`trading/accounts/sample_accounts.yaml`):

```yaml
users:
  - id: diego
    label: Cuenta Diego
    exchanges:
      binance:
        api_key_env: DIEGO_BINANCE_API_KEY
        api_secret_env: DIEGO_BINANCE_API_SECRET
        environment: testnet
  - id: sofia
    label: Cuenta Sofia
    exchanges:
      binance:
        api_key_env: SOFIA_BINANCE_API_KEY
        api_secret_env: SOFIA_BINANCE_API_SECRET
        environment: live
```

Las variables de entorno `DIEGO_BINANCE_API_KEY`, etc., deben estar configuradas en el server (idealmente gestionadas como secretos).

### Uso básico en modo dry-run

```python
from trading.accounts.manager import AccountManager
from trading.orders.executor import OrderExecutor
from trading.orders.models import OrderRequest, OrderSide, OrderType

manager = AccountManager.from_file("accounts.yaml")
executor = OrderExecutor(manager)

order = OrderRequest(
    symbol="ETHUSDT",
    side=OrderSide.BUY,
    type=OrderType.MARKET,
    quantity=0.1,
)

response = executor.execute("diego", "binance", order, dry_run=True)
print(response.status, response.raw)
```

Mientras `dry_run=True` (o la cuenta esté marcada como `testnet`), no se envía la orden a Binance; se devuelve una respuesta simulada. Más adelante se agregará la llamada real a la API, controles de riesgo y manejo de posiciones.

### Configuración de cuentas y secretos

- Cada usuario/exchange hace referencia a variables de entorno (`api_key_env`, `api_secret_env`). En OCI podés declararlas en tu profile, usar un secret manager o exportarlas en el servicio (ej. `export DIEGO_BINANCE_API_KEY=...`).
- Validá rápidamente que todas las cuentas tengan sus claves disponibles:

  ```bash
  python scripts/validate_accounts.py --accounts trading/accounts/sample_accounts.yaml --verbose
  ```

  El script reporta las variables faltantes para que puedas cargarlas antes de habilitar la ejecución real.
- Recordá excluir `accounts.yaml` y `.env` con datos sensibles de tu repo público; mantenelos en el server (o en Vault) y solo referencialos desde las variables de entorno.

## Consejos y buenas prácticas

- Confirmá que `alerts_stream.csv` esté poblado si querés overlay de precios en el dashboard; el watcher `watcher_alertas.py` lo genera automáticamente.
- Para corridas históricas largas, aumentar `PAGINATE_PAGE_LIMIT` y `PAGE_SLEEP_SEC` puede acelerar las descargas sin exceder límites de Binance.
- Si necesitás replicar los resultados en otro equipo, copiá el `.env` (sin credenciales sensibles) y las carpetas `backtestTR/` / `backtestHistorico/`.
- El script maneja comisiones usando el `takerCommissionRate` que expone Binance; si falla la consulta, aplica el fallback 0.0005. Podés forzar una tarifa fija exportando `STRAT_FEE_RATE` antes de ejecutar el backtest.

Con estos pasos deberías poder generar y analizar tanto corridas recientes (TR) como estudios históricos completos de la estrategia.
```

### `backtest/config.py`

```python
"""
Configuración compartida para rutas de backtests.
"""
from __future__ import annotations

import os
from pathlib import Path


OUTPUT_PRESETS: dict[str, dict[str, Path]] = {
    "tr": {
        "trades": Path(os.getenv("STRAT_BACKTEST_TRADES_PATH", "backtest/backtestTR/trades.csv")),
        "plot": Path(os.getenv("STRAT_BACKTEST_PLOT_PATH", "backtest/backtestTR/plot.png")),
        "dashboard": Path(os.getenv("STRAT_BACKTEST_DASHBOARD_PATH", "backtest/backtestTR/dashboard.html")),
    },
    "historico": {
        "trades": Path(os.getenv("STRAT_HIST_BACKTEST_TRADES_PATH", "backtest/backtestHistorico/trades.csv")),
        "plot": Path(os.getenv("STRAT_HIST_BACKTEST_PLOT_PATH", "backtest/backtestHistorico/plot.png")),
        "dashboard": Path(os.getenv("STRAT_HIST_BACKTEST_DASHBOARD_PATH", "backtest/backtestHistorico/dashboard.html")),
    },
}

DEFAULT_PROFILE = os.getenv("BACKTEST_PROFILE", "tr").lower()


def resolve_profile(profile: str | None) -> str:
    candidate = (profile or DEFAULT_PROFILE).lower()
    return candidate if candidate in OUTPUT_PRESETS else "tr"
```

### `backtest/run_backtest.py`

```python
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
STOP_LOSS_PCT = float(os.getenv("STRAT_STOP_LOSS_PCT", "0.05"))
TAKE_PROFIT_PCT = float(os.getenv("STRAT_TAKE_PROFIT_PCT", "0.095"))

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

    ohlc_stream = df_stream[["Open", "High", "Low", "Close", "Volume"]].copy()
    if "CloseTimeDT" in df_stream.columns:
        ohlc_stream["BarCloseTime"] = df_stream["CloseTimeDT"]
    else:
        close_offset = INTERVAL_MS.get(STREAM_INTERVAL, 0)
        ohlc_stream["BarCloseTime"] = df_stream.index + pd.to_timedelta(close_offset, unit="ms")
    bb = compute_bollinger_bands(ohlc_stream, BB_LENGTH, BB_MULT).reindex(ohlc_stream.index).ffill()
    return ohlc_stream, bb


def _compute_risk_levels(direction: str, entry_price: float) -> tuple[float | None, float | None]:
    stop_price = None
    take_price = None
    if entry_price is None or entry_price <= 0:
        return stop_price, take_price

    if STOP_LOSS_PCT > 0:
        if direction == "long":
            stop_price = entry_price * (1 - STOP_LOSS_PCT)
        else:
            stop_price = entry_price * (1 + STOP_LOSS_PCT)

    if TAKE_PROFIT_PCT > 0:
        if direction == "long":
            take_price = entry_price * (1 + TAKE_PROFIT_PCT)
        else:
            take_price = entry_price * (1 - TAKE_PROFIT_PCT)

    return stop_price, take_price


def _check_risk_exit(
    position: dict,
    bar_high: float,
    bar_low: float,
) -> tuple[float, str] | None:
    direction = position["direction"]
    stop_price = position.get("stop_price")
    take_price = position.get("take_price")

    if direction == "long":
        if stop_price is not None and bar_low <= stop_price:
            return float(stop_price), "stop_loss"
        if take_price is not None and bar_high >= take_price:
            return float(take_price), "take_profit"
    else:
        if stop_price is not None and bar_high >= stop_price:
            return float(stop_price), "stop_loss"
        if take_price is not None and bar_low <= take_price:
            return float(take_price), "take_profit"

    return None


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

    ts_open = ohlc.index[row_idx]
    ts_close = ohlc["BarCloseTime"].iloc[row_idx] if "BarCloseTime" in ohlc.columns else ts_open
    ts = ts_close if isinstance(ts_close, pd.Timestamp) else ts_open
    direction_filter = BB_DIRECTION
    basis_now = float(basis.iloc[row_idx]) if basis is not None and not np.isnan(basis.iloc[row_idx]) else np.nan

    if crossed_lower and direction_filter != -1:
        signals.append(
            {
                "type": "bollinger_signal",
                "direction": "long",
                "message": f"{SYMBOL_DISPLAY} {STREAM_INTERVAL}: Señal Bollinger alcista en {lower_now:.2f}",
                "price": lower_now,
                "close_price": close_now,
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
                "close_price": close_now,
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
    pending_entry = None  # orden de entrada límite
    pending_exit = None   # orden de salida límite (cierre por señal opuesta)
    deferred_entry = None # entrada opuesta a ejecutar tras el cierre

    for i in range(1, len(ohlc)):
        ts_open = ohlc.index[i]
        bar_close = float(ohlc["Close"].iloc[i])
        bar_high = float(ohlc["High"].iloc[i])
        bar_low = float(ohlc["Low"].iloc[i])
        ts_close = (
            ohlc["BarCloseTime"].iloc[i]
            if "BarCloseTime" in ohlc.columns
            else ts_open
        )

        # 1) Riesgo (SL / TP). Si se dispara, se cancela todo lo pendiente.
        if position:
            risk_exit = _check_risk_exit(position, bar_high, bar_low)
            if risk_exit:
                exit_price, exit_reason = risk_exit
                position["exit_meta"] = {
                    "basis": position.get("entry_meta", {}).get("basis"),
                    "stop_price": position.get("stop_price"),
                    "take_price": position.get("take_price"),
                }
                trades.append(_finalize_trade(position, float(exit_price), ts_close, exit_reason, fee_rate))
                position = None
                pending_exit = None
                deferred_entry = None

        # 2) Salida límite (banda opuesta) antes de procesar nuevas señales.
        if position and pending_exit:
            exit_price = float(pending_exit["price"])
            direction = position["direction"]
            filled = (direction == "long" and bar_high >= exit_price) or (
                direction == "short" and bar_low <= exit_price
            )
            if filled:
                position["exit_meta"] = {
                    "basis": position.get("entry_meta", {}).get("basis"),
                    "reference_band": pending_exit.get("reference_band"),
                    "stop_price": position.get("stop_price"),
                    "take_price": position.get("take_price"),
                }
                trades.append(_finalize_trade(position, exit_price, ts_close, pending_exit["reason"], fee_rate))
                position = None
                pending_exit = None
                if deferred_entry:
                    pending_entry = deferred_entry
                    deferred_entry = None

        # 3) Entrada límite pendiente si no hay posición
        if pending_entry and position is None:
            entry_price = float(pending_entry["entry_price"])
            direction = pending_entry["direction"]
            filled = (direction == "long" and bar_low <= entry_price) or (
                direction == "short" and bar_high >= entry_price
            )
            if filled:
                entry_ts = ts_close if isinstance(ts_close, pd.Timestamp) else ts_open
                entry_time = pd.Timestamp(entry_ts)
                position = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "entry_time": entry_time,
                    "entry_reason": pending_entry["entry_reason"],
                    "entry_meta": {
                        **pending_entry.get("entry_meta", {}),
                        "order_time": pending_entry.get("order_time"),
                    },
                }
                stop_price, take_price = _compute_risk_levels(direction, entry_price)
                if stop_price is not None:
                    position["stop_price"] = float(stop_price)
                if take_price is not None:
                    position["take_price"] = float(take_price)
                pending_entry = None

        signals = _generate_signal(i, ohlc, bb)
        if not signals:
            continue

        for signal in signals:
            direction = signal["direction"]

            reference = signal.get("reference_band")
            price_base = reference if reference is not None else signal.get("price", bar_close)
            signal_price = float(price_base)
            signal_ts = pd.Timestamp(signal.get("timestamp", ts_open))

            basis_now = signal.get("basis")

            # Si ya hay una posición y la señal es opuesta, programar salida y diferir la entrada opuesta.
            if position:
                if position["direction"] == direction:
                    continue
                pending_exit = {
                    "price": signal_price,
                    "reason": signal["type"],
                    "reference_band": reference,
                }
                deferred_entry = {
                    "direction": direction,
                    "entry_price": signal_price,
                    "entry_reason": signal["type"],
                    "order_time": signal_ts,
                    "entry_meta": {
                        "basis": basis_now,
                        "reference_band": reference,
                    },
                }
                continue

            # Si no hay posición, crear/actualizar la orden de entrada límite.
            entry_price = float(reference) if reference is not None else signal_price
            pending_entry = {
                "direction": direction,
                "entry_price": entry_price,
                "order_time": signal_ts,
                "entry_reason": signal["type"],
                "entry_meta": {
                    "basis": basis_now,
                    "reference_band": reference,
                },
            }

    if position:
        fallback_exit = position["entry_price"]
        position["exit_meta"] = {
            "basis": position.get("entry_meta", {}).get("basis"),
            "stop_price": position.get("stop_price"),
            "take_price": position.get("take_price"),
        }
        final_ts = (
            ohlc["BarCloseTime"].iloc[-1]
            if "BarCloseTime" in ohlc.columns
            else ohlc.index[-1]
        )
        trades.append(_finalize_trade(position, float(fallback_exit), final_ts, "end_of_data", fee_rate))
    if pending_entry:
        pending_entry = None

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

    order_time = entry_meta.get("order_time")
    order_time_ts = pd.Timestamp(order_time) if order_time is not None else entry_time

    fees = (abs(entry_price) + abs(exit_price)) * fee_rate

    log_trade(
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=entry_time,
        order_time=order_time_ts,
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
        order_time_ts.isoformat() if hasattr(order_time_ts, "isoformat") else str(order_time_ts),
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
```

### `backtest/realtime_backtest.py`

```python
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

try:
    from .config import OUTPUT_PRESETS, resolve_profile
    from .run_backtest import _finalize_trade, _fetch_fee_rate, BACKTEST_STREAM_BARS
except ImportError:  # ejecución como script directo
    import sys
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    if str(CURRENT_DIR.parent) not in sys.path:
        sys.path.append(str(CURRENT_DIR.parent))
    from config import OUTPUT_PRESETS, resolve_profile  # type: ignore
    from run_backtest import _finalize_trade, _fetch_fee_rate, BACKTEST_STREAM_BARS  # type: ignore

from velas import API_SYMBOL, STREAM_INTERVAL, compute_bollinger_bands, BB_LENGTH, BB_MULT
from trade_logger import TRADE_COLUMNS
from paginado_binance import fetch_klines_paginado, INTERVAL_MS


BACKTEST_REALTIME_ENABLED = os.getenv("BACKTEST_REALTIME_ENABLED", "true").lower() == "true"
REALTIME_PROFILE = os.getenv("BACKTEST_REALTIME_PROFILE", "tr").lower()
STATE_PATH_ENV = os.getenv("BACKTEST_REALTIME_STATE_PATH", "")
STOP_LOSS_PCT = float(os.getenv("STRAT_STOP_LOSS_PCT", "0.05"))
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

        from .build_dashboard import load_trades as _load_trades_df

        trades_df = _load_trades_df(trades_path)
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
    last_signal_direction = (state or {}).get("last_signal_direction")
    state_status = (state or {}).get("status")
    if state is not None and state_status not in {"pending", "open"}:
        state_status = None

    direction = signal.get("direction")
    if not direction:
        return
    if last_signal_direction == direction:
        return

    ts_raw = signal.get("timestamp")
    signal_ts = _ensure_timestamp(ts_raw)
    reference_band = signal.get("reference_band")
    close_raw = reference_band if reference_band is not None else signal.get("price")
    try:
        order_price = float(close_raw)
    except Exception:
        order_price = float(signal.get("price", 0.0))

    basis_now = signal.get("basis")
    signal_type = signal.get("type", "unknown_signal")

    fee_rate = _fee_rate()

    if state and state_status == "open":
        if state.get("direction") == direction:
            return
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
            exit_price = float(reference_band) if reference_band is not None else order_price
            row = _finalize_trade(position, exit_price, signal_ts, signal_type, fee_rate)
            _append_trade_row(trades_path, row)
        except Exception as exc:
            print(f"[REALTIME][WARN] No se pudo cerrar la posición previa ({exc})")
    if state and state_status == "pending" and state.get("direction") == direction:
        return
    if state and state_status == "pending" and state.get("direction") != direction:
        state = None

    new_state = {
        "status": "pending",
        "direction": direction,
        "entry_price": order_price,
        "order_time": signal_ts.isoformat(),
        "entry_reason": signal_type,
        "entry_meta": {
            "basis": basis_now,
            "reference_band": reference_band,
        },
        "last_signal_direction": direction,
    }
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
    status = state.get("status")
    if status not in {"pending", "open"}:
        return

    direction = state.get("direction")
    entry_price_val = state.get("entry_price", 0.0)
    try:
        entry_price = float(entry_price_val)
    except Exception:
        entry_price = 0.0

    if direction not in {"long", "short"} or entry_price <= 0:
        return

    fee_rate = _fee_rate()

    last_signal_direction = state.get("last_signal_direction", direction)

    if status == "pending":
        order_time_raw = state.get("order_time") or state.get("entry_time")
        if not order_time_raw:
            return
        order_time = _ensure_timestamp(order_time_raw)

        for idx, row in ohlc_stream.iterrows():
            ts_close = row.get("BarCloseTime", idx)
            ts_close_ts = _ensure_timestamp(ts_close)
            if ts_close_ts <= order_time:
                continue
            bar_low = float(row["Low"])
            bar_high = float(row["High"])
            filled = False
            if direction == "long" and bar_low <= entry_price:
                filled = True
            elif direction == "short" and bar_high >= entry_price:
                filled = True
            if filled:
                entry_time = ts_close_ts
                stop_price, take_price = _compute_risk_levels(direction, entry_price)
                new_state = {
                    "status": "open",
                    "direction": direction,
                    "entry_price": entry_price,
                    "entry_time": entry_time.isoformat(),
                    "entry_reason": state.get("entry_reason", "signal"),
                    "entry_meta": {
                        **(state.get("entry_meta") or {}),
                        "order_time": state.get("order_time"),
                    },
                    "last_signal_direction": last_signal_direction,
                }
                if stop_price is not None:
                    new_state["stop_price"] = float(stop_price)
                if take_price is not None:
                    new_state["take_price"] = float(take_price)
                _save_state(new_state, trades_path)
                break
        return

    stop_price = state.get("stop_price")
    take_price = state.get("take_price")
    entry_time_raw = state.get("entry_time")
    if not entry_time_raw:
        return
    entry_time = _ensure_timestamp(entry_time_raw)

    position_data = ohlc_stream.loc[entry_time:]
    if position_data.empty:
        return

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
                "entry_time": entry_time,
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
            _save_state({"last_signal_direction": last_signal_direction}, trades_path)
            if trades_path.exists():
                try:
                    _rebuild_dashboard(resolved_profile, trades_path)
                    _refresh_plot(trades_path)
                except Exception:
                    pass
            break
```

### `backtest/build_dashboard.py`

```python
# build_dashboard.py
import argparse
import html
import os
import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from velas import SYMBOL_DISPLAY, STREAM_INTERVAL
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    if str(PARENT_DIR) not in sys.path:
        sys.path.append(str(PARENT_DIR))
    from velas import SYMBOL_DISPLAY, STREAM_INTERVAL

try:
    from .config import OUTPUT_PRESETS, resolve_profile
except ImportError:  # ejecución directa
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    if str(CURRENT_DIR.parent) not in sys.path:
        sys.path.append(str(CURRENT_DIR.parent))
    from config import OUTPUT_PRESETS, resolve_profile

DEFAULT_PRICE_PATH = Path(os.getenv("ALERTS_TABLE_CSV_PATH", "alerts_stream.csv"))

LOGO_SVG = """
<svg width=\"72\" height=\"72\" viewBox=\"0 0 120 120\" xmlns=\"http://www.w3.org/2000/svg\">
  <rect x=\"0\" y=\"0\" width=\"120\" height=\"120\" rx=\"18\" fill=\"#111827\" stroke=\"#2563eb\" stroke-width=\"6\"/>
  <path d=\"M15 60 C30 40, 55 20, 80 45 S115 100, 105 105\" stroke=\"#22d3ee\" stroke-width=\"6\" fill=\"none\"/>
  <path d=\"M15 80 C40 65, 65 50, 90 70\" stroke=\"#a855f7\" stroke-width=\"6\" fill=\"none\" opacity=\"0.8\"/>
  <circle cx=\"78\" cy=\"46\" r=\"8\" fill=\"#facc15\" stroke=\"#facc15\"/>
</svg>
"""


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de trades: {path}")
    df = pd.read_csv(path)
    if "EntryTime" in df.columns:
        df["EntryTime"] = pd.to_datetime(df["EntryTime"])
    if "ExitTime" in df.columns:
        df["ExitTime"] = pd.to_datetime(df["ExitTime"])
    if "OrderTime" in df.columns:
        df["OrderTime"] = pd.to_datetime(df["OrderTime"])
    else:
        df["OrderTime"] = df.get("EntryTime")
    return df


def load_price(path: Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    if not path.exists():
        print(f"[DASHBOARD][WARN] Archivo de precios no encontrado: {path}")
        return None
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df.set_index("Timestamp", inplace=True)
    return df


def summarize_trades(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"Total trades": 0}
    wins = (df["Outcome"] == "win").sum()
    losses = (df["Outcome"] == "loss").sum()
    total = len(df)
    pnl_pct_sum = df["PnLPct"].sum() * 100
    pnl_pct_avg = df["PnLPct"].mean() * 100
    winrate = wins / total * 100 if total else 0
    cum = df["PnLPct"].fillna(0).cumsum()
    max_drawdown = cum.min() * 100
    total_fees = df.get("Fees", pd.Series(dtype=float)).sum()
    return {
        "Total trades": total,
        "Wins": wins,
        "Losses": losses,
        "Win rate %": f"{winrate:.2f}",
        "Total PnL %": f"{pnl_pct_sum:.2f}",
        "Avg PnL %": f"{pnl_pct_avg:.2f}",
        "Max Drawdown %": f"{max_drawdown:.2f}",
        "Total Fees": f"{total_fees:.2f}",
    }


def build_figure(trades: pd.DataFrame, price_df: pd.DataFrame | None):
    if trades.empty:
        raise ValueError("No hay trades para mostrar.")

    rows = 3 if price_df is not None else 2
    specs = [[{"type": "xy"}] for _ in range(rows)]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        specs=specs,
        row_heights=[0.5, 0.3, 0.2] if rows == 3 else [0.6, 0.4],
    )

    row_idx = 1
    if price_df is not None:
        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=price_df["Close"],
                name="Close",
                line=dict(color="#222", width=1.2),
                hovertemplate="%{x}<br>Close: %{y:.2f}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )

        long_entries = trades[trades["Direction"] == "long"]
        short_entries = trades[trades["Direction"] == "short"]

        if not long_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_entries["EntryTime"],
                    y=long_entries["EntryPrice"],
                    mode="markers",
                    name="Long Entry",
                    marker=dict(symbol="triangle-up", color="#16a34a", size=9),
                    hovertemplate="Long Entry<br>%{x}<br>%{y:.2f}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=long_entries["ExitTime"],
                    y=long_entries["ExitPrice"],
                    mode="markers",
                    name="Long Exit",
                    marker=dict(symbol="x", color="#16a34a", size=9),
                    hovertemplate="Long Exit<br>%{x}<br>%{y:.2f}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

        if not short_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_entries["EntryTime"],
                    y=short_entries["EntryPrice"],
                    mode="markers",
                    name="Short Entry",
                    marker=dict(symbol="triangle-down", color="#dc2626", size=9),
                    hovertemplate="Short Entry<br>%{x}<br>%{y:.2f}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=short_entries["ExitTime"],
                    y=short_entries["ExitPrice"],
                    mode="markers",
                    name="Short Exit",
                    marker=dict(symbol="x", color="#dc2626", size=9),
                    hovertemplate="Short Exit<br>%{x}<br>%{y:.2f}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

        row_idx += 1

    cum_pct = trades["PnLPct"].fillna(0).cumsum() * 100
    fig.add_trace(
        go.Scatter(
            x=trades["ExitTime"],
            y=cum_pct,
            mode="lines+markers",
            name="PnL acumulado %",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=6),
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
        ),
        row=row_idx,
        col=1,
    )
    fig.update_yaxes(title_text="PnL %", row=row_idx, col=1)
    row_idx += 1

    fig.add_trace(
        go.Histogram(
            x=trades["PnLPct"] * 100,
            nbinsx=20,
            marker=dict(color="#737373"),
            name="Distribución PnL %",
            hovertemplate="%{x:.2f}%<extra></extra>",
        ),
        row=row_idx,
        col=1,
    )
    fig.update_yaxes(title_text="Frecuencia", row=row_idx, col=1)
    fig.update_xaxes(title_text="PnL (%)", row=row_idx, col=1)

    fig.update_layout(
        height=780,
        template="plotly_white",
        title="Dashboard Estrategia Bollinger",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def build_summary_html(summary: dict) -> str:
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary.items())
    return f"""
    <section class="summary">
        <h2>Resumen</h2>
        <table>
            {rows}
        </table>
    </section>
    """


def _fmt_two(value, *, blank: str = "") -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return blank
    if pd.isna(num):
        return blank
    return f"{num:.2f}"


def _fmt_pct(value, *, blank: str = "") -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return blank
    if pd.isna(num):
        return blank
    return f"{num * 100:.5f}%"


def _fmt_timestamp(value, fmt: str) -> str:
    if value is None:
        return ""
    try:
        ts = pd.Timestamp(value)
    except Exception:
        text = str(value)
        return "" if not text or text.lower() in {"nat", "nan"} else text
    if pd.isna(ts):
        return ""
    return ts.strftime(fmt)


def _safe_text(value, *, blank: str = "") -> str:
    if value is None:
        return blank
    if isinstance(value, float) and pd.isna(value):
        return blank
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat"}:
        return blank
    return text


def _normalize_value(value) -> str:
    text = _safe_text(value, blank="")
    return text.lower()


def _data_attr_name(key: str) -> str:
    parts: list[str] = []
    for ch in key:
        if ch.isupper():
            parts.append("-")
            parts.append(ch.lower())
        else:
            parts.append(ch)
    return "data-" + "".join(parts)


def build_trades_table(trades: pd.DataFrame) -> str:
    columns = [
        ("EntryTime", "Entrada"),
        ("OrderTime", "Orden Banda"),
        ("ExitTime", "Salida"),
        ("Direction", "Dirección"),
        ("EntryReason", "Motivo Entrada"),
        ("ExitReason", "Motivo Salida"),
        ("EntryPrice", "Precio Entrada"),
        ("ExitPrice", "Precio Salida"),
        ("Outcome", "Resultado"),
        ("PnLAbs", "PnL"),
        ("PnLPct", "PnL %"),
        ("Fees", "Fees"),
    ]

    header_cells = "".join(f"<th>{label}</th>" for _, label in columns)
    rows_html: list[str] = []

    for _, row in trades.iterrows():
        attrs = {
            "direction": _normalize_value(row.get("Direction")),
            "entryReason": _normalize_value(row.get("EntryReason")),
            "exitReason": _normalize_value(row.get("ExitReason")),
            "outcome": _normalize_value(row.get("Outcome")),
        }
        attr_parts = [
            f'{_data_attr_name(key)}="{html.escape(value)}"'
            for key, value in attrs.items()
            if value
        ]
        attr_str = f" {' '.join(attr_parts)}" if attr_parts else ""

        cells: list[str] = []
        direction_raw = _safe_text(row.get("Direction"), blank="")
        direction_norm = direction_raw.lower() if direction_raw else ""
        outcome_raw = _safe_text(row.get("Outcome"), blank="")
        outcome_norm = outcome_raw.lower() if outcome_raw else ""

        for key, _ in columns:
            if key in {"EntryTime", "OrderTime", "ExitTime"}:
                text = _fmt_timestamp(row.get(key), "%Y-%m-%d %H:%M:%S")
                cells.append(f"<td>{html.escape(text)}</td>")
            elif key in {"EntryPrice", "ExitPrice", "PnLAbs", "Fees"}:
                text = _fmt_two(row.get(key), blank="")
                cells.append(f"<td>{html.escape(text)}</td>")
            elif key == "PnLPct":
                text = _fmt_pct(row.get(key), blank="")
                cells.append(f"<td>{html.escape(text)}</td>")
            elif key == "Direction":
                if direction_raw:
                    cells.append(
                        f"<td class='dir {direction_norm}'>{html.escape(direction_raw.upper())}</td>"
                    )
                else:
                    cells.append("<td></td>")
            elif key == "Outcome":
                if outcome_raw:
                    cells.append(
                        f"<td class='result {outcome_norm}'>{html.escape(outcome_raw.upper())}</td>"
                    )
                else:
                    cells.append("<td></td>")
            else:
                text = _safe_text(row.get(key), blank="")
                cells.append(f"<td>{html.escape(text)}</td>")

        rows_html.append(f"<tr{attr_str}>{''.join(cells)}</tr>")

    body_rows = "".join(rows_html)
    return f"""
    <section class="trades">
        <h2>Todos los trades</h2>
        <table class="trades-table filterable">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{body_rows}</tbody>
        </table>
    </section>
    """


def build_operations_table(trades: pd.DataFrame, limit: int = 15) -> str:
    columns = [
        ("EntryTime", "Entrada"),
        ("OrderTime", "Orden Banda"),
        ("Direction", "Dirección"),
        ("EntryReason", "Motivo Entrada"),
        ("ExitReason", "Motivo Salida"),
        ("ExitTime", "Salida"),
        ("EntryPrice", "Precio Entrada"),
        ("ExitPrice", "Precio Salida"),
        ("Outcome", "Resultado"),
        ("PnLAbs", "PnL"),
        ("PnLPct", "PnL %"),
        ("Fees", "Fees"),
    ]

    subset = trades.tail(limit)
    header_cells = "".join(f"<th>{label}</th>" for _, label in columns)
    rows_html: list[str] = []

    for _, row in subset.iterrows():
        attrs = {
            "direction": _normalize_value(row.get("Direction")),
            "entryReason": _normalize_value(row.get("EntryReason")),
            "exitReason": _normalize_value(row.get("ExitReason")),
            "outcome": _normalize_value(row.get("Outcome")),
        }
        attr_parts = [
            f'{_data_attr_name(key)}="{html.escape(value)}"'
            for key, value in attrs.items()
            if value
        ]
        attr_str = f" {' '.join(attr_parts)}" if attr_parts else ""

        direction_raw = _safe_text(row.get("Direction"), blank="")
        direction_norm = direction_raw.lower() if direction_raw else ""
        outcome_raw = _safe_text(row.get("Outcome"), blank="")
        outcome_norm = outcome_raw.lower() if outcome_raw else ""

        cells: list[str] = []
        for key, _ in columns:
            if key in {"EntryTime", "OrderTime", "ExitTime"}:
                text = _fmt_timestamp(row.get(key), "%d-%m %H:%M")
                cells.append(f"<td>{html.escape(text or '--')}</td>")
            elif key in {"EntryPrice", "ExitPrice", "PnLAbs", "Fees"}:
                text = _fmt_two(row.get(key), blank="--")
                cells.append(f"<td>{html.escape(text)}</td>")
            elif key == "PnLPct":
                text = _fmt_pct(row.get(key), blank="--")
                cells.append(f"<td>{html.escape(text)}</td>")
            elif key == "Direction":
                if direction_raw:
                    cells.append(
                        f"<td class='dir {direction_norm}'>{html.escape(direction_raw.upper())}</td>"
                    )
                else:
                    cells.append("<td></td>")
            elif key == "Outcome":
                if outcome_raw:
                    cells.append(
                        f"<td class='result {outcome_norm}'>{html.escape(outcome_raw.upper())}</td>"
                    )
                else:
                    cells.append("<td></td>")
            else:
                text = _safe_text(row.get(key), blank="--")
                cells.append(f"<td>{html.escape(text)}</td>")

        rows_html.append(f"<tr{attr_str}>{''.join(cells)}</tr>")

    body_rows = "".join(rows_html)
    return f"""
    <section class="ops">
        <h2>Detalle Operativo Reciente</h2>
        <table class="ops-table filterable">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{body_rows}</tbody>
        </table>
    </section>
    """


def _collect_filter_values(series: pd.Series | None) -> list[tuple[str, str]]:
    if series is None:
        return []
    try:
        iterable = series.dropna().unique()
    except Exception:
        iterable = []

    mapping: dict[str, str] = {}
    for raw in iterable:
        label = _safe_text(raw, blank="")
        norm = _normalize_value(raw)
        if not norm:
            continue
        mapping.setdefault(norm, label)

    return sorted(mapping.items(), key=lambda item: item[1].lower())


def build_filters_html(trades: pd.DataFrame) -> str:
    specs = [
        ("direction", "Dirección", trades["Direction"] if "Direction" in trades.columns else None),
        ("entryReason", "Motivo Entrada", trades["EntryReason"] if "EntryReason" in trades.columns else None),
        ("exitReason", "Motivo Salida", trades["ExitReason"] if "ExitReason" in trades.columns else None),
        ("outcome", "Resultado", trades["Outcome"] if "Outcome" in trades.columns else None),
    ]

    controls = []
    for key, label, series in specs:
        options = _collect_filter_values(series)
        options_html = "".join(
            f"<option value='{html.escape(value)}'>{html.escape(display)}</option>"
            for value, display in options
        )
        control_html = f"""
        <label>
            <span>{html.escape(label)}:</span>
            <select data-filter-key="{key}">
                <option value="">Todos</option>
                {options_html}
            </select>
        </label>
        """
        controls.append(control_html)

    controls_html = "".join(controls)
    return f"""
    <section class="filters">
        <h2>Filtrar trades</h2>
        <div class="filters-grid">
            {controls_html}
        </div>
    </section>
    """


def render_dashboard(trades_path: Path, price_path: Path | None, html_out: Path, show: bool, profile: str):
    trades_df = load_trades(trades_path)
    price_df = load_price(price_path)
    summary = summarize_trades(trades_df)

    print("[DASHBOARD] Resumen trades:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    fig = build_figure(trades_df, price_df)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False})

    summary_html = build_summary_html(summary)
    filters_html = build_filters_html(trades_df)
    ops_table_html = build_operations_table(trades_df)
    trades_table_html = build_trades_table(trades_df)

    html_out.parent.mkdir(parents=True, exist_ok=True)

    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8" />
    <title>Dashboard Estrategia Bollinger</title>
    <style>
        body {{
            font-family: 'Inter', Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 32px 24px 56px;
        }}
        .hero {{
            display: flex;
            align-items: center;
            gap: 24px;
            margin-bottom: 24px;
        }}
        .hero .logo {{
            flex-shrink: 0;
        }}
        .hero h1 {{
            margin: 0;
            font-size: 1.9rem;
        }}
        .hero p {{
            margin: 6px 0 0;
            color: #94a3b8;
        }}
        h2 {{
            margin-top: 32px;
            border-left: 4px solid #2563eb;
            padding-left: 12px;
            font-size: 1.3rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            background: #1e293b;
            border-radius: 10px;
            overflow: hidden;
        }}
        th, td {{
            padding: 10px 14px;
            border-bottom: 1px solid #334155;
            text-align: left;
            font-size: 0.95rem;
        }}
        th {{
            color: #60a5fa;
            background: rgba(37, 99, 235, 0.12);
        }}
        .trades-table tbody tr:nth-child(even),
        .ops-table tbody tr:nth-child(even) {{
            background: rgba(15, 23, 42, 0.6);
        }}
        .plot-container {{
            margin-top: 32px;
        }}
        .dir.long {{
            color: #22c55e;
            font-weight: 600;
        }}
        .dir.short {{
            color: #f87171;
            font-weight: 600;
        }}
        .result.win {{
            color: #4ade80;
            font-weight: 600;
        }}
        .result.loss {{
            color: #f87171;
            font-weight: 600;
        }}
        .result.flat {{
            color: #fbbf24;
            font-weight: 600;
        }}
        .filters {{
            margin-top: 32px;
            padding: 18px 20px;
            background: #1e293b;
            border-radius: 10px;
        }}
        .filters h2 {{
            margin: 0 0 12px;
            font-size: 1.2rem;
        }}
        .filters-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px 24px;
        }}
        .filters label {{
            display: flex;
            flex-direction: column;
            font-size: 0.9rem;
            color: #cbd5f5;
        }}
        .filters label span {{
            margin-bottom: 6px;
            color: #94a3b8;
            font-weight: 600;
        }}
        .filters select {{
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 6px 10px;
            color: #f8fafc;
            min-width: 160px;
        }}
        .filters select:focus {{
            outline: none;
            border-color: #2563eb;
            box-shadow: 0 0 0 1px #2563eb;
        }}
        @media (max-width: 768px) {{
            .hero {{
                flex-direction: column;
                align-items: flex-start;
            }}
            .hero .logo {{
                margin-bottom: 8px;
            }}
            th, td {{
                font-size: 0.85rem;
            }}
            .filters-grid {{
                flex-direction: column;
            }}
            .filters select {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <section class="hero">
        <div class="logo">{LOGO_SVG}</div>
        <div>
            <h1>Dashboard Estrategia Bollinger</h1>
            <p>{SYMBOL_DISPLAY} · Intervalo {STREAM_INTERVAL} · Perfil {profile.upper()}</p>
        </div>
    </section>
    {summary_html}
    <div class="plot-container">
        {fig_html}
    </div>
    {filters_html}
    {ops_table_html}
    {trades_table_html}
    <script>
    (function() {{
        const selects = document.querySelectorAll('select[data-filter-key]');
        if (!selects.length) {{
            return;
        }}
        const tables = document.querySelectorAll('table.filterable');

        function applyFilters() {{
            const active = {{}};
            selects.forEach((sel) => {{
                const value = sel.value;
                if (value) {{
                    active[sel.dataset.filterKey] = value;
                }}
            }});

            tables.forEach((table) => {{
                table.querySelectorAll('tbody tr').forEach((row) => {{
                    let visible = true;
                    for (const [key, value] of Object.entries(active)) {{
                        const rowValue = (row.dataset[key] || '');
                        if (rowValue !== value) {{
                            visible = false;
                            break;
                        }}
                    }}
                    row.style.display = visible ? '' : 'none';
                }});
            }});
        }}

        selects.forEach((sel) => sel.addEventListener('change', applyFilters));
        applyFilters();
    }})();
    </script>
</body>
</html>"""

    html_out.write_text(full_html, encoding="utf-8")
    print(f"[DASHBOARD] HTML generado en {html_out}")

    if show:
        webbrowser.open(html_out.resolve().as_uri())


def main():
    parser = argparse.ArgumentParser(description="Dashboard HTML para trades de la estrategia Bollinger.")
    parser.add_argument("--profile", choices=sorted(OUTPUT_PRESETS.keys()), default=None, help="Preset de salidas (tr o historico).")
    parser.add_argument("--trades", type=str, default=None, help="CSV con trades a visualizar.")
    parser.add_argument("--price", type=str, default=str(DEFAULT_PRICE_PATH), help="CSV con precios (ej. alerts_stream.csv).")
    parser.add_argument("--html", type=str, default=None, help="Archivo HTML de salida.")
    parser.add_argument("--show", action="store_true", help="Abrir el dashboard en el navegador al finalizar.")
    args = parser.parse_args()

    profile = resolve_profile(args.profile)
    preset_paths = OUTPUT_PRESETS[profile]

    trades_path = Path(args.trades) if args.trades else preset_paths["trades"]
    price_path = Path(args.price) if args.price else None
    html_path = Path(args.html) if args.html else preset_paths["dashboard"]

    render_dashboard(trades_path, price_path, html_path, args.show, profile)


if __name__ == "__main__":
    main()
```

### `backtest/order_fill_listener.py`

```python
# order_fill_listener.py
"""
Listener dedicado que monitorea órdenes pendientes y registra
el minuto exacto en que se ejecutan según velas de 1 minuto.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from binance.um_futures import UMFutures
from dotenv import load_dotenv
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from .config import OUTPUT_PRESETS, resolve_profile
    from .realtime_backtest import (
        _compute_risk_levels,
        _ensure_timestamp,
        _load_state,
        _save_state,
        _state_path,
    )
except ImportError:  # ejecución directa fuera del paquete
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    if str(PARENT_DIR) not in sys.path:
        sys.path.append(str(PARENT_DIR))
    from config import OUTPUT_PRESETS, resolve_profile
    from realtime_backtest import (  # type: ignore
        _compute_risk_levels,
        _ensure_timestamp,
        _load_state,
        _save_state,
        _state_path,
    )


def _um_client() -> UMFutures:
    base_url = os.getenv("BINANCE_UM_BASE_URL", "https://fapi.binance.com")
    return UMFutures(base_url=base_url)


def _symbol_from_env() -> str:
    symbol = os.getenv("SYMBOL", "ETHUSDT.P")
    return symbol.replace(".P", "")


def _detect_fill(
    client: UMFutures,
    *,
    symbol: str,
    direction: str,
    entry_price: float,
    order_time: pd.Timestamp,
    tolerance: float,
    lookback_minutes: int,
) -> Optional[pd.Timestamp]:
    """
    Devuelve el cierre de la primera vela de 1 minuto que toca el precio objetivo.
    Si aún no se ejecutó, retorna None.
    """
    if direction not in {"long", "short"} or entry_price <= 0:
        return None

    order_time_utc = order_time.tz_convert("UTC")
    start_ts = order_time_utc - pd.Timedelta(minutes=lookback_minutes)
    start_ms = int(max(0, start_ts.timestamp() * 1000))
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)

    try:
        klines = client.klines(
            symbol=symbol,
            interval="1m",
            startTime=start_ms,
            endTime=now_ms,
            limit=1000,
        )
    except Exception as exc:
        print(f"[LISTENER][WARN] No se pudieron descargar velas 1m ({exc})")
        return None

    if not klines:
        return None

    tolerance_abs = abs(tolerance)
    target_up = entry_price + tolerance_abs
    target_down = entry_price - tolerance_abs

    for candle in klines:
        open_ms = int(candle[0])
        close_ms = int(candle[6])
        open_ts = pd.Timestamp(open_ms, unit="ms", tz="UTC")
        close_ts = pd.Timestamp(close_ms, unit="ms", tz="UTC")

        if close_ts <= order_time_utc:
            continue

        high = float(candle[2])
        low = float(candle[3])

        if direction == "long":
            if low <= target_up:
                return close_ts
        else:
            if high >= target_down:
                return close_ts

    return None


def _transition_to_open(
    *,
    trades_path: Path,
    state: dict,
    fill_timestamp: pd.Timestamp,
) -> None:
    direction = state.get("direction")
    entry_price = float(state["entry_price"])
    entry_reason = state.get("entry_reason", "signal")
    entry_meta = state.get("entry_meta") or {}
    last_signal = state.get("last_signal_direction", direction)

    stop_price, take_price = _compute_risk_levels(direction, entry_price)
    new_state = {
        "status": "open",
        "direction": direction,
        "entry_price": entry_price,
        "entry_time": fill_timestamp.isoformat(),
        "entry_reason": entry_reason,
        "entry_meta": {
            **entry_meta,
            "order_time": state.get("order_time"),
        },
        "last_signal_direction": last_signal,
    }
    if stop_price is not None:
        new_state["stop_price"] = float(stop_price)
    if take_price is not None:
        new_state["take_price"] = float(take_price)

    _save_state(new_state, trades_path)
    print(f"[LISTENER] Orden '{direction}' ejecutada en {fill_timestamp.isoformat()}")


def run_listener(
    *,
    profile: str,
    poll_seconds: float,
    tolerance: float,
    lookback_minutes: int,
) -> None:
    trades_path = OUTPUT_PRESETS[profile]["trades"]
    state_path = _state_path(trades_path)
    print(f"[LISTENER] Usando perfil {profile} | State: {state_path}")

    client = _um_client()
    symbol = _symbol_from_env()
    tz_name = os.getenv("TZ", "UTC")
    try:
        local_tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        local_tz = ZoneInfo("UTC")

    while True:
        state = _load_state(trades_path)
        if not state:
            time.sleep(poll_seconds)
            continue

        status = state.get("status")
        if status != "pending":
            time.sleep(poll_seconds)
            continue

        try:
            direction = state["direction"]
            entry_price = float(state["entry_price"])
            order_time_raw = state.get("order_time") or state.get("entry_time")
        except (KeyError, ValueError, TypeError):
            time.sleep(poll_seconds)
            continue

        if not order_time_raw:
            time.sleep(poll_seconds)
            continue

        order_time = _ensure_timestamp(order_time_raw)
        fill_ts = _detect_fill(
            client,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            order_time=order_time,
            tolerance=tolerance,
            lookback_minutes=lookback_minutes,
        )

        if fill_ts is None:
            time.sleep(poll_seconds)
            continue

        try:
            fill_ts_local = fill_ts.tz_convert(local_tz)
        except Exception:
            fill_ts_local = fill_ts

        _transition_to_open(
            trades_path=trades_path,
            state=state,
            fill_timestamp=fill_ts_local,
        )
        time.sleep(poll_seconds)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Listener dedicado para captar el minuto exacto de ejecución de órdenes pendientes."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(OUTPUT_PRESETS.keys()),
        default=None,
        help="Perfil de salidas a monitorear (tr/historico).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.getenv("ORDER_LISTENER_POLL_SECONDS", "15")),
        help="Segundos de espera entre chequeos.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=float(os.getenv("ORDER_LISTENER_PRICE_TOL", "0.0")),
        help="Tolerancia absoluta en el match de precio.",
    )
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=int(os.getenv("ORDER_LISTENER_LOOKBACK_MINUTES", "120")),
        help="Minutos hacia atrás que se consultan al buscar la vela que llenó la orden.",
    )

    args = parser.parse_args()
    profile = resolve_profile(args.profile)

    run_listener(
        profile=profile,
        poll_seconds=max(1.0, args.poll_seconds),
        tolerance=args.tolerance,
        lookback_minutes=max(1, args.lookback_minutes),
    )


if __name__ == "__main__":
    main()
```

### `scripts/dashcrud.py`

```python
#!/usr/bin/env python3
"""
DashCRUD: dashboard mínimo para CRUD de cuentas/exchanges.

- Usa YAML/JSON en trading/accounts/* para persistir.
- No expone secretos; solo nombres de variables de entorno.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse, unquote
import shutil
import subprocess

import requests
import yaml

# Asegura imports relativos al repo
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading.accounts.manager import AccountManager
from trading.accounts.models import AccountConfig, ExchangeCredential, ExchangeEnvironment

DEFAULT_ACCOUNTS_PATH = Path("trading/accounts/oci_accounts.yaml")
DEFAULT_HTML = REPO_ROOT / "trading/accounts/dashcrud.html"
DEFAULT_ENV_PATH = Path(os.getenv("DASHCRUD_ENV_PATH", "/home/ubuntu/bot/.env"))
# Path secundario opcional para compatibilidad (systemd env file)
SECONDARY_ENV_PATH = Path("/etc/systemd/system/bot.env")
FALLBACK_SYMBOLS = {"binance": {"ETHUSDT", "BTCUSDT"}, "bybit": {"ETHUSDT", "BTCUSDT"}}


def _normalize_identifier(value: str) -> str:
    """Normaliza IDs para usarlos en nombres de variables (sin espacios)."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    # Compacta guiones/underscores múltiples
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("_-")


def _load_manager(path: Path) -> AccountManager:
    try:
        return AccountManager.from_file(path)
    except FileNotFoundError:
        print(f"[WARN] {path} no existe; se inicializa vacío.")
        return AccountManager.empty()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] No se pudo leer {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _serialize(manager: AccountManager, accounts_path: Path) -> dict:
    data = manager.to_dict()
    out = []
    for acc in data.get("users", []):
        exchanges = []
        for name, cred in (acc.get("exchanges") or {}).items():
            exchanges.append(
                {
                    "name": name,
                    "exchange": cred.get("exchange", name),
                    "environment": cred.get("environment"),
                    "api_key_env": cred.get("api_key_env"),
                    "api_secret_env": cred.get("api_secret_env"),
                    "notional_usdt": cred.get("notional_usdt"),
                    "leverage": cred.get("leverage"),
                    "symbol": (cred.get("extra") or {}).get("symbol"),
                    "extra": cred.get("extra") or {},
                }
            )
        out.append(
            {
                "id": acc["id"],
                "label": acc.get("label", acc["id"]),
                "enabled": bool(acc.get("enabled", True)),
                "metadata": acc.get("metadata") or {},
                "exchanges": exchanges,
            }
        )
    return {"accounts_path": str(accounts_path), "users": out}


def _validate_symbol(exchange: str, environment: ExchangeEnvironment, symbol: str) -> None:
    """
    Valida el símbolo contra el exchange.
    - Binance: consulta exchangeInfo.
    - dYdX, Bybit: se acepta sin validar contra la API (asumimos símbolo válido), con fallback si se configuró.
    - Otros: usa fallback si está configurado.
    """
    ex = exchange.lower()
    sym = symbol.upper()
    if ex == "binance":
        base_url = "https://testnet.binancefuture.com" if environment == ExchangeEnvironment.TESTNET else "https://fapi.binance.com"
        try:
            resp = requests.get(f"{base_url}/fapi/v1/exchangeInfo", timeout=8)
            resp.raise_for_status()
            data = resp.json()
            symbols = {
                s["symbol"]
                for s in data.get("symbols", [])
                if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL"
            }
            if sym not in symbols:
                raise ValueError(f"El símbolo {sym} no está disponible en {exchange} ({environment.value}).")
            return
        except requests.RequestException as exc:
            if sym in FALLBACK_SYMBOLS.get(ex, set()):
                return
            raise ValueError(f"No se pudo validar el símbolo en {exchange}: {exc}")
    if ex in {"dydx", "bybit"}:
        return
    if sym in FALLBACK_SYMBOLS.get(ex, set()):
        return
    raise ValueError(f"No se reconoce el exchange '{exchange}' o el símbolo {sym} no está permitido.")


def _generate_env_names(user_id: str, exchange: str, environment: ExchangeEnvironment) -> tuple[str, str]:
    normalized_user = _normalize_identifier(user_id)
    base = f"{normalized_user}_{exchange}_{environment.value}".upper().replace("-", "_")
    return f"{base}_API_KEY", f"{base}_API_SECRET"


def _load_env_file(env_path: Path) -> list[str]:
    if not env_path.exists():
        return []
    try:
        return env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []


def _save_env_file(env_path: Path, lines: list[str]) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    env_path.write_text(content, encoding="utf-8")


def _set_env_vars(env_path: Path, mapping: Dict[str, str]) -> None:
    """Actualiza/crea variables en el env file, con backup previo."""
    try:
        lines = _load_env_file(env_path)
        out = []
        seen = set()
        for line in lines:
            if not line or line.strip().startswith("#") or "=" not in line:
                out.append(line)
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if key in mapping:
                out.append(f"{key}={mapping[key]}")
                seen.add(key)
            else:
                out.append(line)
        # add missing keys
        for k, v in mapping.items():
            if k not in seen:
                out.append(f"{k}={v}")
        _save_env_file(env_path, out)
    except PermissionError as exc:
        print(f"[ENV][WARN] No se pudo escribir {env_path}: {exc}; las variables deben cargarse a mano.")
        return


def _build_credential(payload: Dict[str, Any], default_name: str | None = None, *, user_id: str | None = None, env_path: Path = DEFAULT_ENV_PATH) -> ExchangeCredential:
    name = (payload.get("exchange") or payload.get("name") or default_name or "").lower()
    if not name:
        raise ValueError("exchange es obligatorio.")
    env_raw = (payload.get("environment") or ExchangeEnvironment.TESTNET.value).lower()
    try:
        environment = ExchangeEnvironment(env_raw)
    except ValueError:
        valid = [e.value for e in ExchangeEnvironment]
        raise ValueError(f"environment debe ser uno de {valid}.")

    api_key_env = str(payload.get("api_key_env") or "").strip()
    api_secret_env = str(payload.get("api_secret_env") or "").strip()
    # Valores en texto plano (acepta *_plain o *_text)
    api_key_plain = str(
        payload.get("api_key_plain")
        or payload.get("api_key_text")
        or payload.get("api_key")
        or ""
    ).strip()
    api_secret_plain = str(
        payload.get("api_secret_plain")
        or payload.get("api_secret_text")
        or payload.get("api_secret")
        or ""
    ).strip()
    def _looks_like_secret(val: str) -> bool:
        return bool(val) and len(val) >= 20 and " " not in val and "=" not in val

    # Heurística: si el usuario pegó las claves en los campos *_env (confusión común), las tratamos como valores.
    if not api_key_plain and not api_secret_plain and _looks_like_secret(api_key_env) and _looks_like_secret(api_secret_env):
        api_key_plain, api_secret_plain = api_key_env, api_secret_env
        api_key_env, api_secret_env = "", ""

    if api_key_plain and api_secret_plain:
        if not user_id:
            raise ValueError("user_id es obligatorio para generar variables de entorno.")
        # Si no especificaron nombres de env, se generan; si los pasaron, se usan esos
        if not api_key_env or not api_secret_env:
            gen_key, gen_secret = _generate_env_names(user_id, name, environment)
            api_key_env, api_secret_env = gen_key, gen_secret
        _set_env_vars(env_path, {api_key_env: api_key_plain, api_secret_env: api_secret_plain})
        # Compatibilidad: también intentamos escribir en /etc/systemd/system/bot.env si existe o se puede
        try:
            _set_env_vars(SECONDARY_ENV_PATH, {api_key_env: api_key_plain, api_secret_env: api_secret_plain})
        except Exception:
            pass

    if not api_key_env or not api_secret_env:
        raise ValueError("api_key_env y api_secret_env son obligatorios (o provée keys en texto para generarlas).")

    symbol = str(payload.get("symbol") or payload.get("pair") or "").strip().upper()
    if not symbol and name == "dydx":
        symbol = "ETH-USD"
    if not symbol:
        raise ValueError("symbol es obligatorio.")

    _validate_symbol(name, environment, symbol)

    notional_val = payload.get("notional_usdt")
    leverage_val = payload.get("leverage")
    notional = float(notional_val) if notional_val not in (None, "", False) else None
    leverage = int(leverage_val) if leverage_val not in (None, "", False) else None
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    extra = {**extra, "symbol": symbol}
    if name == "dydx" and "subaccount" not in extra:
        extra["subaccount"] = int(payload.get("subaccount") or 0)

    return ExchangeCredential(
        exchange=name,
        api_key_env=api_key_env,
        api_secret_env=api_secret_env,
        environment=environment,
        notional_usdt=notional,
        leverage=leverage,
        extra=extra,
    )


def _save_with_backup(manager: AccountManager, path: Path) -> None:
    if path.exists():
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak.{ts}")
        shutil.copy2(path, backup)
    manager.save_to_file(path)


def _restart_services(services: list[str]) -> tuple[bool, str | None]:
    """
    Reinicia servicios systemd. Devuelve (ok, error_msg).
    """
    cmds = [
        ["sudo", "systemctl", "restart", *services],
        ["systemctl", "restart", *services],
    ]
    errors = []
    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True)
            return True, None
        except subprocess.CalledProcessError as exc:  # pragma: no cover - externo
            errors.append(str(exc))
    return False, "; ".join(errors)


class DashCRUDHandler(BaseHTTPRequestHandler):
    manager: AccountManager
    accounts_path: Path
    html_path: Path

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        """Log mínimo a stdout."""
        msg = fmt % args
        print(f"[HTTP] {self.address_string()} {msg}")

    # --- Helpers -------------------------------------------------- #
    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            try:
                decoded = raw.decode("utf-8", errors="replace")
                print(f"[HTTP][WARN] JSON inválido (json.loads): {decoded}")
                # Intento con YAML para tolerar pequeños desvíos de sintaxis
                try:
                    alt = yaml.safe_load(decoded)
                    if isinstance(alt, dict):
                        print("[HTTP][INFO] JSON parseado vía YAML fallback")
                        return alt
                except Exception as exc:
                    print(f"[HTTP][WARN] YAML fallback falló: {exc}")
            except Exception:
                pass
            self._send_json(400, {"error": "JSON inválido"})
            return None

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self) -> None:
        try:
            html = self.html_path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, "Dashboard HTML no encontrado")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _snapshot(self) -> dict:
        return _serialize(self.manager, self.accounts_path)

    # --- Routing -------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        parts = [unquote(p) for p in path.split("/") if p]

        if path == "/":
            self._serve_html()
            return
        if parts[:2] == ["api", "accounts"]:
            if len(parts) == 2:
                self._send_json(200, self._snapshot())
                return
            if len(parts) == 3:
                user_id = parts[2]
                try:
                    acc = self.manager.get_account(user_id)
                except KeyError:
                    self._send_json(404, {"error": f"No existe la cuenta '{user_id}'."})
                    return
                tmp_manager = AccountManager([acc])
                self._send_json(200, _serialize(tmp_manager, self.accounts_path))
                return
        self.send_error(404, "Ruta no encontrada")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        if parts == ["api", "accounts"]:
            payload = self._read_json()
            if payload is None:
                return
            user_id = (payload.get("id") or payload.get("user_id") or "").strip()
            label = (payload.get("label") or "").strip()
            enabled = bool(payload.get("enabled", True))
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            if not user_id:
                self._send_json(400, {"error": "id es obligatorio."})
                return
            if user_id in {a.user_id for a in self.manager.list_accounts()}:
                self._send_json(409, {"error": f"La cuenta '{user_id}' ya existe."})
                return
            account = self.manager.upsert_account(user_id, label=label or None, metadata=metadata, enabled=enabled)
            exchange_payload = payload.get("exchange") or {}
            if exchange_payload:
                try:
                    cred = _build_credential(exchange_payload, user_id=user_id, env_path=DEFAULT_ENV_PATH)
                    self.manager.upsert_exchange(user_id, cred)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
            _save_with_backup(self.manager, self.accounts_path)
            tmp_manager = AccountManager([account])
            self._send_json(201, _serialize(tmp_manager, self.accounts_path))
            return

        self.send_error(404, "Ruta no encontrada")

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        if parts[:2] != ["api", "accounts"]:
            self.send_error(404, "Ruta no encontrada")
            return
        payload = self._read_json()
        if payload is None:
            return

        if len(parts) == 3:
            # Actualiza cuenta (label/enabled/metadata)
            user_id = parts[2]
            try:
                self.manager.get_account(user_id)
            except KeyError:
                self._send_json(404, {"error": f"No existe la cuenta '{user_id}'."})
                return
            new_id = (payload.get("id") or payload.get("user_id") or user_id).strip()
            if not new_id:
                self._send_json(400, {"error": "id no puede ser vacío."})
                return
            if new_id != user_id and new_id in {a.user_id for a in self.manager.list_accounts()}:
                self._send_json(409, {"error": f"La cuenta '{new_id}' ya existe."})
                return

            label = payload.get("label")
            enabled = payload.get("enabled")
            metadata = payload.get("metadata")
            if new_id != user_id:
                self.manager.rename_account(user_id, new_id)
                user_id = new_id
            self.manager.upsert_account(
                user_id,
                label=label if label is not None else None,
                metadata=metadata if isinstance(metadata, dict) else None,
                enabled=enabled if enabled is not None else None,
            )
            _save_with_backup(self.manager, self.accounts_path)
            self._send_json(200, self._snapshot())
            return

        if len(parts) == 4 and parts[3] == "exchange":
            user_id = parts[2]
            try:
                account = self.manager.get_account(user_id)
            except KeyError:
                self._send_json(404, {"error": f"No existe la cuenta '{user_id}'."})
                return
            try:
                cred = _build_credential(payload, default_name=payload.get("exchange") or payload.get("name"), user_id=user_id, env_path=DEFAULT_ENV_PATH)
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"error": str(exc)})
                return
            # Mantener un único exchange por usuario: se limpia y se inserta el nuevo.
            account.exchanges = {}
            self.manager.upsert_exchange(user_id, cred)
            _save_with_backup(self.manager, self.accounts_path)
            ok, err = _restart_services(["bot-watcher.service", "bot-strategy.service", "bot-order-listener.service"])
            resp = self._snapshot()
            if not ok and err:
                resp["warning"] = f"No se pudieron reiniciar servicios: {err}"
            self._send_json(200, resp)
            return

        self.send_error(404, "Ruta no encontrada")

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        if parts[:2] != ["api", "accounts"] or len(parts) != 3:
            self.send_error(404, "Ruta no encontrada")
            return
        user_id = parts[2]
        try:
            account = self.manager.get_account(user_id)
        except KeyError:
            self._send_json(404, {"error": f"No existe la cuenta '{user_id}'."})
            return
        # Borrado lógico: enabled = False
        account.enabled = False
        _save_with_backup(self.manager, self.accounts_path)
        self._send_json(200, self._snapshot())


def _build_handler(manager: AccountManager, accounts_path: Path, html_path: Path):
    class _Handler(DashCRUDHandler):
        pass

    _Handler.manager = manager
    _Handler.accounts_path = accounts_path
    _Handler.html_path = html_path
    return _Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="DashCRUD: server HTTP para cuentas/exchanges.")
    parser.add_argument("--accounts", type=str, default=DEFAULT_ACCOUNTS_PATH, help="Archivo YAML/JSON de cuentas.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host de escucha (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8050, help="Puerto de escucha (default 8050).")
    parser.add_argument("--html", type=str, default=None, help="Ruta del HTML del dashboard.")
    args = parser.parse_args()

    accounts_path = Path(args.accounts)
    html_path = Path(args.html) if args.html else DEFAULT_HTML
    manager = _load_manager(accounts_path)
    handler = _build_handler(manager, accounts_path, html_path)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[INFO] DashCRUD en http://{args.host}:{args.port}")
    print(f"[INFO] Archivo de cuentas: {accounts_path}")
    print(f"[INFO] HTML: {html_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Detenido por el usuario.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### `scripts/test_dydx_diego.py`

```python
#!/usr/bin/env python3
"""
Script de prueba para verificar la configuración de dYdX para el usuario Diego.
"""
import os
import sys
from pathlib import Path

# Asegura imports relativos al repo
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading.accounts.manager import AccountManager
from trading.orders.executor import OrderExecutor
from trading.orders.models import OrderRequest, OrderSide, OrderType, TimeInForce

def test_dydx_config():
    """Prueba la configuración de dYdX para Diego."""
    print("=" * 60)
    print("Test de Configuración dYdX - Usuario Diego")
    print("=" * 60)
    
    # 1. Cargar configuración
    accounts_file = Path("trading/accounts/oci_accounts.yaml")
    if not accounts_file.exists():
        print(f"[ERROR] No se encontró {accounts_file}")
        return 1
    
    try:
        manager = AccountManager.from_file(accounts_file)
        print(f"[OK] Configuración cargada desde {accounts_file}")
    except Exception as exc:
        print(f"[ERROR] No se pudo cargar configuración: {exc}")
        return 1
    
    # 2. Verificar cuenta Diego
    try:
        account = manager.get_account("diego")
        print(f"[OK] Cuenta Diego encontrada: {account.label}")
    except KeyError:
        print("[ERROR] No se encontró la cuenta 'diego'")
        return 1
    
    # 3. Verificar exchange dYdX
    try:
        cred = account.get_exchange("dydx")
        print(f"[OK] Exchange dYdX configurado para Diego")
        print(f"    - Environment: {cred.environment.value}")
        print(f"    - Symbol: {cred.extra.get('symbol', 'N/A')}")
        print(f"    - Subaccount: {cred.extra.get('subaccount', 'N/A')}")
        print(f"    - Notional USDC: {cred.notional_usdc}")
        print(f"    - Max Position USDC: {cred.max_position_usdc}")
        print(f"    - Margin Mode: {cred.margin_mode}")
    except KeyError:
        print("[ERROR] Diego no tiene configuración para dYdX")
        return 1
    
    # 4. Verificar variables de entorno
    print("\n[INFO] Verificando variables de entorno...")
    api_key_env = cred.api_key_env
    api_secret_env = cred.api_secret_env
    
    api_key = os.getenv(api_key_env)
    api_secret = os.getenv(api_secret_env)
    
    if not api_key:
        print(f"[ERROR] Variable {api_key_env} no está definida")
        return 1
    else:
        print(f"[OK] {api_key_env} = {api_key[:10]}...{api_key[-10:]}")
    
    if not api_secret:
        print(f"[ERROR] Variable {api_secret_env} no está definida")
        return 1
    else:
        masked_secret = api_secret[:6] + "..." + api_secret[-6:] if len(api_secret) > 12 else "***"
        print(f"[OK] {api_secret_env} = {masked_secret}")
    
    # 5. Verificar formato de credenciales
    print("\n[INFO] Verificando formato de credenciales...")
    if api_key.startswith("dydx1"):
        print(f"[OK] API Key tiene formato wallet address (dydx1...)")
    else:
        print(f"[WARN] API Key no tiene formato wallet address esperado (dydx1...)")
        print(f"       Esto puede indicar que es una API key tradicional (legacy)")
    
    if api_secret.startswith("0x"):
        print(f"[OK] API Secret tiene formato private key (0x...)")
    elif len(api_secret) == 64:
        print(f"[OK] API Secret tiene formato hex (64 caracteres)")
    else:
        print(f"[WARN] API Secret no tiene formato private key esperado")
        print(f"       Esto puede indicar que es un API secret tradicional (legacy)")
    
    # 6. Probar OrderExecutor (dry-run)
    print("\n[INFO] Probando OrderExecutor (dry-run)...")
    try:
        executor = OrderExecutor(manager)
        order = OrderRequest(
            symbol=cred.extra.get("symbol", "ETH-USD"),
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=0.01,
            time_in_force=TimeInForce.GTC,
        )
        
        response = executor.execute("diego", "dydx", order, dry_run=True)
        if response.success:
            print(f"[OK] OrderExecutor funciona correctamente (dry-run)")
            print(f"    - Status: {response.status}")
            print(f"    - Symbol: {order.symbol}")
            print(f"    - Side: {order.side.value}")
            print(f"    - Quantity: {order.quantity}")
        else:
            print(f"[ERROR] OrderExecutor falló: {response.error}")
            return 1
    except Exception as exc:
        print(f"[ERROR] Error probando OrderExecutor: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 7. Verificar variables opcionales
    print("\n[INFO] Verificando variables opcionales...")
    grpc_host = os.getenv("DYDX_GRPC_HOST")
    if grpc_host:
        print(f"[OK] DYDX_GRPC_HOST = {grpc_host}")
    else:
        print(f"[INFO] DYDX_GRPC_HOST no definida (usará default: dydx-dao-grpc-1.polkachu.com:443)")
    
    indexer_url = os.getenv("DYDX_INDEXER_URL")
    if indexer_url:
        print(f"[OK] DYDX_INDEXER_URL = {indexer_url}")
    else:
        print(f"[INFO] DYDX_INDEXER_URL no definida (usará default)")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Configuración de dYdX para Diego está completa")
    print("=" * 60)
    print("\nNotas:")
    print("- Las credenciales están configuradas correctamente")
    print("- OrderExecutor funciona en modo dry-run")
    print("- Para trading real, asegúrate de:")
    print("  1. Tener fondos en la wallet de dYdX")
    print("  2. Configurar WATCHER_ENABLE_TRADING=true")
    print("  3. Configurar WATCHER_TRADING_DRY_RUN=false")
    print("  4. Verificar que el símbolo ETH-USD esté disponible en dYdX")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_dydx_config())
```

### `scripts/validate_accounts.py`

```python
#!/usr/bin/env python3
"""
Valida que las cuentas configuradas tengan sus credenciales en variables de entorno.

Uso:
    python scripts/validate_accounts.py --accounts trading/accounts/sample_accounts.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from trading.accounts.manager import AccountManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida credenciales definidas en accounts.yaml.")
    parser.add_argument(
        "--accounts",
        type=str,
        default=os.getenv("TRADING_ACCOUNTS_FILE", "trading/accounts/sample_accounts.yaml"),
        help="Ruta al archivo YAML/JSON con la configuración de cuentas.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Imprime el detalle completo de cada cuenta.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.accounts)
    try:
        manager = AccountManager.from_file(path)
    except Exception as exc:
        print(f"[ERROR] No se pudo leer {path}: {exc}", file=sys.stderr)
        return 1

    missing = []
    for account in manager.list_accounts():
        if args.verbose:
            print(f"- Cuenta: {account.user_id} ({account.label})")
        for name, credential in account.exchanges.items():
            try:
                credential.resolve_keys(os.environ)
                if args.verbose:
                    envs = (credential.api_key_env, credential.api_secret_env)
                    print(f"  • {name}: OK ({envs[0]}, {envs[1]})")
            except RuntimeError as exc:
                missing.append(str(exc))
                if args.verbose:
                    print(f"  • {name}: ERROR -> {exc}")

    if missing:
        print("\n[WARN] Credenciales faltantes:")
        for msg in missing:
            print(f"  - {msg}")
        print("\nExportá las variables correspondientes (ej: export VAR=valor) o configuralas en tu gestor de secretos.")
        return 2

    print(f"[OK] Todas las cuentas de {path} tienen credenciales disponibles en las variables de entorno.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
