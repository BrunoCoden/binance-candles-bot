from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from .models import AccountConfig, ExchangeCredential, ExchangeEnvironment

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


class AccountManager:
    """
    Administra múltiples cuentas/usuarios y sus credenciales por exchange.
    """

    def __init__(self, accounts: Iterable[AccountConfig]):
        self._accounts: Dict[str, AccountConfig] = {}
        for account in accounts:
            self._accounts[account.user_id] = account

    @classmethod
    def empty(cls) -> "AccountManager":
        return cls(accounts=[])

    @classmethod
    def from_dict(cls, data: Mapping) -> "AccountManager":
        users = data.get("users") or data.get("accounts") or []
        accounts: list[AccountConfig] = []
        for entry in users:
            user_id = entry.get("id") or entry.get("user_id")
            if not user_id:
                raise ValueError("Cada cuenta debe especificar 'id'.")
            label = entry.get("label") or user_id
            exchanges_data = entry.get("exchanges") or {}
            exchanges: Dict[str, ExchangeCredential] = {}
            for ex_name, ex_conf in exchanges_data.items():
                exchange = ex_conf.get("exchange", ex_name).lower()
                env_value = (ex_conf.get("environment") or ExchangeEnvironment.TESTNET.value).lower()
                environment = ExchangeEnvironment(env_value)
                cred = ExchangeCredential(
                    exchange=exchange,
                    api_key_env=ex_conf["api_key_env"],
                    api_secret_env=ex_conf["api_secret_env"],
                    environment=environment,
                    extra={k: v for k, v in ex_conf.items() if k not in {"api_key_env", "api_secret_env", "environment"}},
                )
                exchanges[exchange] = cred
            metadata = entry.get("metadata") or {}
            accounts.append(
                AccountConfig(
                    user_id=user_id,
                    label=label,
                    exchanges=exchanges,
                    metadata=metadata,
                )
            )
        return cls(accounts)

    @classmethod
    def from_file(cls, path: Path) -> "AccountManager":
        if not path.exists():
            raise FileNotFoundError(f"No se encontró archivo de cuentas: {path}")

        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML no está instalado. `pip install pyyaml` para leer archivos YAML.")
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        else:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        return cls.from_dict(data)

    def list_accounts(self) -> list[AccountConfig]:
        return list(self._accounts.values())

    def get_account(self, user_id: str) -> AccountConfig:
        try:
            return self._accounts[user_id]
        except KeyError as exc:
            raise KeyError(f"No existe la cuenta '{user_id}'.") from exc

    def get_exchange_credential(self, user_id: str, exchange: str) -> ExchangeCredential:
        account = self.get_account(user_id)
        return account.get_exchange(exchange)

    def resolve_keys(self, user_id: str, exchange: str, env: Optional[Mapping[str, str]] = None) -> tuple[str, str]:
        credential = self.get_exchange_credential(user_id, exchange)
        env_mapping = env or os.environ
        return credential.resolve_keys(env_mapping)

    def to_dict(self) -> dict:
        return {
            "users": [
                {
                    **asdict(acc),
                    "exchanges": {name: asdict(cred) for name, cred in acc.exchanges.items()},
                }
                for acc in self._accounts.values()
            ]
        }
