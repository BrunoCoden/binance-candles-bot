"""
Registro de exchanges soportados y sus clientes.
"""

from .base import ExchangeClient, ExchangeRegistry

from . import binance  # noqa: F401  # Registro de cliente Binance por defecto
from . import dydx  # noqa: F401  # Registro de cliente dYdX

__all__ = [
    "ExchangeClient",
    "ExchangeRegistry",
]
