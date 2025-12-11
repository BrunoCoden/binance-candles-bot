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
        # Forzamos post-only en entradas para evitar fills a otro precio.
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
            skip_bracket = bool(order.extra_params.get("skip_bracket")) if order.extra_params else False
            bracket_raw: Dict[str, Any] = {}
            # Solo gestionar TP/SL si corresponde: no lo hacemos en reduceOnly, ni cuando se pide skip_bracket.
            if (tp or sl) and not order.reduce_only and not skip_bracket:
                try:
                    # Cancela brackets previos y ajusta qty al tamaño esperado de la posición tras esta orden.
                    canceled = self._cancel_open_reduce_only(client, order.symbol)
                    current_pos = self._current_position_qty(client, order.symbol)
                    signed_qty = order.quantity if order.side.value.upper() == "BUY" else -order.quantity
                    expected_pos = current_pos + signed_qty
                    bracket_qty = abs(expected_pos) if expected_pos != 0 else order.quantity
                    if bracket_qty <= 0:
                        bracket_qty = order.quantity
                    bracket_raw = {
                        "canceled": canceled,
                        **self._place_bracket(client, order.symbol, order.side.value, bracket_qty, tp, sl),
                    }
                except Exception as exc:  # pragma: no cover - externo
                    logger.error("Error enviando TP/SL: %s", exc)
                    bracket_raw = {"bracket_error": str(exc)}
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
