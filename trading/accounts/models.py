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
    # dYdX v4 soporte: passphrase/stark keys y notional USDC / max position
    passphrase_env: str | None = None
    stark_key_env: str | None = None
    notional_usdc: float | None = None
    max_position_usdc: float | None = None
    margin_mode: str | None = None  # 'isolated' o 'cross' (dYdX soporta isolated markets)

    def resolve_keys(self, env: Mapping[str, str]) -> tuple[str, str]:
        api_key = env.get(self.api_key_env)
        api_secret = env.get(self.api_secret_env)
        if (
            not api_key
            or not api_secret
            or api_key.strip() == self.api_key_env
            or api_secret.strip() == self.api_secret_env
        ):
            raise RuntimeError(
                f"Credenciales faltantes para {self.exchange}: "
                f"{self.api_key_env}/{self.api_secret_env}"
            )
        return api_key, api_secret

    def resolve_optional(self, env: Mapping[str, str], key: str | None) -> str | None:
        if key is None:
            return None
        return env.get(key) or None


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
