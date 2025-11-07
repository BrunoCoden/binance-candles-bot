from __future__ import annotations

from typing import Dict

from .base import ExchangeClient, ExchangeRegistry
from ..accounts.models import AccountConfig, ExchangeCredential, ExchangeEnvironment
from ..orders.models import CancelRequest, CancelResponse, OrderRequest, OrderResponse
from ..utils.logging import get_logger

logger = get_logger("trading.exchanges.binance")


class BinanceClient(ExchangeClient):
    name = "binance"

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

        if dry_run or credential.environment == ExchangeEnvironment.TESTNET:
            # Simulación: no se llama a la API, solo se devuelve una respuesta mock
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

        # Placeholder para implementación futura (orden real)
        logger.warning(
            "La ejecución real todavía no está implementada. Configurá dry_run=True o usa testnet."
        )
        return OrderResponse(success=False, status="UNSUPPORTED", error="Live trading not implemented")

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
