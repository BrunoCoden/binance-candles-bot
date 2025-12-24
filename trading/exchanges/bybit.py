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
        # pybit v5 unified trading; allow override of domain.
        domain_env = os.getenv("BYBIT_DOMAIN_TESTNET" if is_testnet else "BYBIT_DOMAIN")
        if not domain_env:
            domain_env = "api-demo.bybit.com" if is_testnet else "api.bybit.com"
        # Cuando se provee domain, evitamos el prefijo interno de testnet para usar el host exacto.
        return HTTP(api_key=api_key, api_secret=api_secret, testnet=False, domain=domain_env)

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
