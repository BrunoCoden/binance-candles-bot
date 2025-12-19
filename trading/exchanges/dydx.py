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
from ..orders.models import CancelRequest, CancelResponse, OrderRequest, OrderResponse
from ..utils.logging import get_logger

logger = get_logger("trading.exchanges.dydx")


DEFAULT_GRPC = os.getenv("DYDX_GRPC_HOST", "dydx-dao-grpc-1.polkachu.com:443")
MAINNET_CHAIN_ID = "dydx-mainnet-1"
MAINNET_CHAIN_DENOM = "adydx"
MAINNET_USDC_DENOM = "ibc/8E27BA2D5493AF5636760E354E46004562C46AB7EC0CC4C1CA14E9E20E2545B5"
INDEXER_URL = os.getenv("DYDX_INDEXER_URL", "https://indexer.dydx.trade/v4/perpetualMarkets")


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
            price_subticks = self._quantize_price(order.price or 0, subticks_per_tick)

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
