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

    def resolve_keys(self, env: Mapping[str, str]) -> tuple[str, str]:
        api_key = env.get(self.api_key_env)
        api_secret = env.get(self.api_secret_env)
        if not api_key or not api_secret:
            raise RuntimeError(
                f"Credenciales faltantes para {self.exchange}: "
                f"{self.api_key_env}/{self.api_secret_env}"
            )
        return api_key, api_secret


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
