from __future__ import annotations

import os
from typing import Any, Dict, Optional, List
from decimal import Decimal, ROUND_DOWN

from bybit import bybit

from .base import ExchangeClient, ExchangeRegistry
from ..accounts.models import AccountConfig, ExchangeCredential, ExchangeEnvironment
from ..orders.models import OrderRequest, OrderResponse, CancelRequest, CancelResponse
from ..utils.logging import get_logger

logger = get_logger("trading.exchanges.bybit")


class BybitClient(ExchangeClient):
    name = "bybit"

    def _build_client(self, credential: ExchangeCredential):
        api_key, api_secret = credential.resolve_keys(os.environ)
        # SDK 0.2.x infiere los endpoints estándar con el flag test (True=testnet, False=live).
        # No acepta parámetro domain, por lo que usamos la ruta por defecto del SDK.
        return bybit(
            test=credential.environment != ExchangeEnvironment.LIVE,
            api_key=api_key,
            api_secret=api_secret,
        )

    @staticmethod
    def _quantize(value: float, step: str) -> str:
        dv = Decimal(str(value)).quantize(Decimal(step), rounding=ROUND_DOWN)
        if dv <= 0:
            dv = Decimal(step)
        return format(dv, "f")

    def _format_order_params(self, order: OrderRequest) -> Dict[str, Any]:
        # Bybit USDT perpetual (linear). Ajustar symbol si se parametriza.
        params: Dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side.value.capitalize(),  # BUY/SELL
            "order_type": order.type.value.capitalize(),  # MARKET/LIMIT
            "qty": self._quantize(order.quantity, "0.001"),
            "reduce_only": order.reduce_only,
            "time_in_force": "GoodTillCancel",  # por defecto; Bybit ignora TIF en MARKET
        }
        if order.type.value == "LIMIT" and order.price:
            params["price"] = self._quantize(order.price, "0.1")
            if not order.reduce_only:
                params["order_link_id"] = order.client_order_id or None
        elif order.type.value == "MARKET":
            # Bybit no requiere price/TIF en MARKET; no mandar postOnly
            params.pop("time_in_force", None)
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
            resp = client.Order.Order_new(**params).result()
            raw = resp[0] if isinstance(resp, tuple) else resp
            order_id = ""
            status = "NEW"
            if isinstance(raw, dict):
                order_id = str(raw.get("order_id") or raw.get("result", {}).get("order_id") or "")
                status = raw.get("status") or raw.get("ret_msg") or "NEW"
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
            resp = client.Order.Order_cancel(symbol=request.symbol, order_id=request.exchange_order_id).result()
            raw = resp[0] if isinstance(resp, tuple) else resp
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
            resp = client.Wallet.Wallet_getBalance(coin="USDT").result()
            raw = resp[0] if isinstance(resp, tuple) else resp
            if isinstance(raw, dict):
                bal = raw.get("result", {}).get("USDT", {}).get("wallet_balance")
                return {"USDT": float(bal) if bal is not None else 0.0}
        except Exception:  # pragma: no cover - externo
            return {"USDT": 0.0}
        return {"USDT": 0.0}


ExchangeRegistry.register(BybitClient)
