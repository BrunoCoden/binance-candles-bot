from __future__ import annotations

import os
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict

from dydx_v4_client.client import Client as DydxClient
from dydx_v4_client.constants import ORDER_SIDE_BUY, ORDER_SIDE_SELL, ORDER_TYPE_LIMIT, ORDER_TYPE_MARKET

from .base import ExchangeClient, ExchangeRegistry
from ..accounts.models import AccountConfig, ExchangeCredential, ExchangeEnvironment
from ..orders.models import CancelRequest, CancelResponse, OrderRequest, OrderResponse
from ..utils.logging import get_logger

logger = get_logger("trading.exchanges.dydx")


class DydxClientWrapper(ExchangeClient):
    name = "dydx"

    def _build_client(self, credential: ExchangeCredential) -> DydxClient:
        api_key, api_secret = credential.resolve_keys(os.environ)
        passphrase = credential.resolve_optional(os.environ, credential.passphrase_env)
        stark_key = credential.resolve_optional(os.environ, credential.stark_key_env)
        host = "https://api.dydx.exchange" if credential.environment == ExchangeEnvironment.LIVE else "https://testnet.dydx.exchange"
        kwargs: Dict[str, Any] = {
            "api_key": api_key,  # API wallet address (permissioned key)
            "api_secret": api_secret,  # private key (permissioned)
            "host": host,
            "subaccount_number": 0,
        }
        if passphrase:
            kwargs["passphrase"] = passphrase
        if stark_key:
            kwargs["stark_private_key"] = stark_key
        return DydxClient(**kwargs)

    @staticmethod
    def _quantize(value: float, step: str) -> str:
        dv = Decimal(str(value)).quantize(Decimal(step), rounding=ROUND_DOWN)
        if dv <= 0:
            dv = Decimal(step)
        return format(dv, "f")

    def _market_meta(self, client: DydxClient, symbol: str) -> dict:
        try:
            markets = client.public.get_markets().get("markets", {})
            return markets.get(symbol, {})
        except Exception as exc:  # pragma: no cover - externo
            logger.error("No se pudo obtener mercados dYdX: %s", exc)
            return {}

    def _place_order_params(self, order: OrderRequest, market: dict, credential: ExchangeCredential) -> Dict[str, Any]:
        tick_size = market.get("tickSize") or "0.1"
        step_size = market.get("stepSize") or "0.001"
        side = ORDER_SIDE_BUY if order.side.value.upper() == "BUY" else ORDER_SIDE_SELL
        order_type = ORDER_TYPE_LIMIT if order.type.value.upper() == "LIMIT" else ORDER_TYPE_MARKET
        params: Dict[str, Any] = {
            "market": order.symbol,
            "side": side,
            "type": order_type,
            "size": self._quantize(order.quantity, step_size),
            "reduceOnly": bool(order.reduce_only),
        }
        if order_type == ORDER_TYPE_LIMIT:
            params["price"] = self._quantize(order.price or 0, tick_size)
            params["timeInForce"] = "POSTONLY"
            params["postOnly"] = True
        margin_mode = credential.margin_mode or "isolated"
        params["marginMode"] = margin_mode
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
            "dYdX place_order dry_run=%s user=%s symbol=%s side=%s qty=%s type=%s price=%s",
            dry_run,
            account.user_id,
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
        client = self._build_client(credential)
        market = self._market_meta(client, order.symbol)
        params = self._place_order_params(order, market, credential)
        try:
            resp = client.private.create_order(**params)
            oid = resp.get("order").get("id") if isinstance(resp, dict) else None
            status = resp.get("order").get("status") if isinstance(resp, dict) else "NEW"
            return OrderResponse(
                success=True,
                status=status or "NEW",
                exchange_order_id=str(oid) if oid else None,
                filled_quantity=float(order.quantity),
                avg_price=float(order.price or 0.0),
                raw=resp,
            )
        except Exception as exc:
            logger.exception("dYdX error enviando orden: %s", exc)
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
            resp = client.private.cancel_order(order_id=request.exchange_order_id, market=request.symbol)
            return CancelResponse(success=True, raw=resp)
        except Exception as exc:
            logger.exception("dYdX error cancelando orden: %s", exc)
            return CancelResponse(success=False, raw={}, error=str(exc))

    def fetch_account_balance(
        self,
        account: AccountConfig,
        credential: ExchangeCredential,
    ) -> Dict[str, float]:
        try:
            client = self._build_client(credential)
            acc = client.private.get_account()
            equity = acc.get("account", {}).get("equity") if isinstance(acc, dict) else 0
            return {"USDC": float(equity or 0)}
        except Exception:
            return {"USDC": 0.0}


ExchangeRegistry.register(DydxClientWrapper)
