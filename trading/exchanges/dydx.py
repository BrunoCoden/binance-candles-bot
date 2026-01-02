from __future__ import annotations

"""
Cliente dYdX v4 (NodeClient).
- Usa endpoint gRPC seguro público (mainnet).
- Soporta múltiples usuarios (AccountManager).
- Envía órdenes LIMIT/POST-ONLY en quantums/subticks según clob_pair.
"""

import os
import asyncio
import re
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
        if r.status_code == 404:
            # El indexer devuelve 404 cuando no hay datos para esa address/subaccount (p.ej. nunca operó).
            return 0.0
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
        private_clean = (private_hex or "").strip()
        private_clean = private_clean[2:] if private_clean.lower().startswith("0x") else private_clean
        if not private_clean:
            raise ValueError("dYdX: api_secret vacío (se espera private key hex).")
        if not re.fullmatch(r"[0-9a-fA-F]+", private_clean):
            raise ValueError(
                "dYdX: api_secret inválido; se espera hex (opcionalmente con prefijo 0x)."
            )
        if len(private_clean) % 2 == 1:
            logger.warning("dYdX: api_secret hex con longitud impar; zero-padding para from_hex().")
            private_clean = "0" + private_clean
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
