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
