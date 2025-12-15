#!/usr/bin/env python3
"""
DashCRUD: dashboard mínimo para CRUD de cuentas/exchanges.

- Usa YAML/JSON en trading/accounts/* para persistir.
- No expone secretos; solo nombres de variables de entorno.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse
import shutil

import requests

# Asegura imports relativos al repo
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading.accounts.manager import AccountManager
from trading.accounts.models import AccountConfig, ExchangeCredential, ExchangeEnvironment

DEFAULT_ACCOUNTS_PATH = Path("trading/accounts/oci_accounts.yaml")
DEFAULT_HTML = REPO_ROOT / "trading/accounts/dashcrud.html"
DEFAULT_ENV_PATH = Path(os.getenv("DASHCRUD_ENV_PATH", "/etc/systemd/system/bot.env"))
FALLBACK_SYMBOLS = {"binance": {"ETHUSDT", "BTCUSDT"}}


def _load_manager(path: Path) -> AccountManager:
    try:
        return AccountManager.from_file(path)
    except FileNotFoundError:
        print(f"[WARN] {path} no existe; se inicializa vacío.")
        return AccountManager.empty()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] No se pudo leer {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _serialize(manager: AccountManager, accounts_path: Path) -> dict:
    data = manager.to_dict()
    out = []
    for acc in data.get("users", []):
        exchanges = []
        for name, cred in (acc.get("exchanges") or {}).items():
            exchanges.append(
                {
                    "name": name,
                    "exchange": cred.get("exchange", name),
                    "environment": cred.get("environment"),
                    "api_key_env": cred.get("api_key_env"),
                    "api_secret_env": cred.get("api_secret_env"),
                    "notional_usdt": cred.get("notional_usdt"),
                    "leverage": cred.get("leverage"),
                    "symbol": (cred.get("extra") or {}).get("symbol"),
                    "extra": cred.get("extra") or {},
                }
            )
        out.append(
            {
                "id": acc["id"],
                "label": acc.get("label", acc["id"]),
                "enabled": bool(acc.get("enabled", True)),
                "metadata": acc.get("metadata") or {},
                "exchanges": exchanges,
            }
        )
    return {"accounts_path": str(accounts_path), "users": out}


def _validate_symbol(exchange: str, environment: ExchangeEnvironment, symbol: str) -> None:
    """
    Valida el símbolo contra el exchange.
    - Binance: consulta exchangeInfo.
    - dYdX: se acepta sin validar contra la API (testnet/live difieren; se asume símbolo válido).
    - Otros: usa fallback si está configurado.
    """
    ex = exchange.lower()
    sym = symbol.upper()
    if ex == "binance":
        base_url = "https://testnet.binancefuture.com" if environment == ExchangeEnvironment.TESTNET else "https://fapi.binance.com"
        try:
            resp = requests.get(f"{base_url}/fapi/v1/exchangeInfo", timeout=8)
            resp.raise_for_status()
            data = resp.json()
            symbols = {
                s["symbol"]
                for s in data.get("symbols", [])
                if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL"
            }
            if sym not in symbols:
                raise ValueError(f"El símbolo {sym} no está disponible en {exchange} ({environment.value}).")
            return
        except requests.RequestException as exc:
            if sym in FALLBACK_SYMBOLS.get(ex, set()):
                return
            raise ValueError(f"No se pudo validar el símbolo en {exchange}: {exc}")
    if ex == "dydx":
        # No se valida aquí; se asume símbolo dYdX válido.
        return
    if sym in FALLBACK_SYMBOLS.get(ex, set()):
        return
    raise ValueError(f"No se reconoce el exchange '{exchange}' o el símbolo {sym} no está permitido.")


def _generate_env_names(user_id: str, exchange: str, environment: ExchangeEnvironment) -> tuple[str, str]:
    base = f"{user_id}_{exchange}_{environment.value}".upper().replace("-", "_")
    return f"{base}_API_KEY", f"{base}_API_SECRET"


def _load_env_file(env_path: Path) -> list[str]:
    if not env_path.exists():
        return []
    try:
        return env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []


def _save_env_file(env_path: Path, lines: list[str]) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    env_path.write_text(content, encoding="utf-8")


def _set_env_vars(env_path: Path, mapping: Dict[str, str]) -> None:
    """Actualiza/crea variables en el env file, con backup previo."""
    if env_path.exists():
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        shutil.copy2(env_path, env_path.with_suffix(env_path.suffix + f".bak.{ts}"))

    lines = _load_env_file(env_path)
    out = []
    seen = set()
    for line in lines:
        if not line or line.strip().startswith("#") or "=" not in line:
            out.append(line)
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key in mapping:
            out.append(f"{key}={mapping[key]}")
            seen.add(key)
        else:
            out.append(line)
    # add missing keys
    for k, v in mapping.items():
        if k not in seen:
            out.append(f"{k}={v}")
    _save_env_file(env_path, out)


def _build_credential(payload: Dict[str, Any], default_name: str | None = None, *, user_id: str | None = None, env_path: Path = DEFAULT_ENV_PATH) -> ExchangeCredential:
    name = (payload.get("exchange") or payload.get("name") or default_name or "").lower()
    if not name:
        raise ValueError("exchange es obligatorio.")
    env_raw = (payload.get("environment") or ExchangeEnvironment.TESTNET.value).lower()
    try:
        environment = ExchangeEnvironment(env_raw)
    except ValueError:
        valid = [e.value for e in ExchangeEnvironment]
        raise ValueError(f"environment debe ser uno de {valid}.")

    api_key_env = (payload.get("api_key_env") or "").strip()
    api_secret_env = (payload.get("api_secret_env") or "").strip()
    api_key_plain = (payload.get("api_key_plain") or "").strip()
    api_secret_plain = (payload.get("api_secret_plain") or "").strip()
    if api_key_plain and api_secret_plain:
        if not user_id:
            raise ValueError("user_id es obligatorio para generar variables de entorno.")
        gen_key, gen_secret = _generate_env_names(user_id, name, environment)
        _set_env_vars(env_path, {gen_key: api_key_plain, gen_secret: api_secret_plain})
        api_key_env, api_secret_env = gen_key, gen_secret

    if not api_key_env or not api_secret_env:
        raise ValueError("api_key_env y api_secret_env son obligatorios (o provée keys en texto para generarlas).")

    symbol = (payload.get("symbol") or payload.get("pair") or "").strip().upper()
    if not symbol:
        raise ValueError("symbol es obligatorio.")

    _validate_symbol(name, environment, symbol)

    notional_val = payload.get("notional_usdt")
    leverage_val = payload.get("leverage")
    notional = float(notional_val) if notional_val not in (None, "", False) else None
    leverage = int(leverage_val) if leverage_val not in (None, "", False) else None
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    extra = {**extra, "symbol": symbol}

    return ExchangeCredential(
        exchange=name,
        api_key_env=api_key_env,
        api_secret_env=api_secret_env,
        environment=environment,
        notional_usdt=notional,
        leverage=leverage,
        extra=extra,
    )


def _save_with_backup(manager: AccountManager, path: Path) -> None:
    if path.exists():
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak.{ts}")
        shutil.copy2(path, backup)
    manager.save_to_file(path)


class DashCRUDHandler(BaseHTTPRequestHandler):
    manager: AccountManager
    accounts_path: Path
    html_path: Path

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        """Log mínimo a stdout."""
        msg = fmt % args
        print(f"[HTTP] {self.address_string()} {msg}")

    # --- Helpers -------------------------------------------------- #
    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "JSON inválido"})
            return None

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self) -> None:
        try:
            html = self.html_path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, "Dashboard HTML no encontrado")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _snapshot(self) -> dict:
        return _serialize(self.manager, self.accounts_path)

    # --- Routing -------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        parts = [p for p in path.split("/") if p]

        if path == "/":
            self._serve_html()
            return
        if parts[:2] == ["api", "accounts"]:
            if len(parts) == 2:
                self._send_json(200, self._snapshot())
                return
            if len(parts) == 3:
                user_id = parts[2]
                try:
                    acc = self.manager.get_account(user_id)
                except KeyError:
                    self._send_json(404, {"error": f"No existe la cuenta '{user_id}'."})
                    return
                tmp_manager = AccountManager([acc])
                self._send_json(200, _serialize(tmp_manager, self.accounts_path))
                return
        self.send_error(404, "Ruta no encontrada")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if parts == ["api", "accounts"]:
            payload = self._read_json()
            if payload is None:
                return
            user_id = (payload.get("id") or payload.get("user_id") or "").strip()
            label = (payload.get("label") or "").strip()
            enabled = bool(payload.get("enabled", True))
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            if not user_id:
                self._send_json(400, {"error": "id es obligatorio."})
                return
            if user_id in {a.user_id for a in self.manager.list_accounts()}:
                self._send_json(409, {"error": f"La cuenta '{user_id}' ya existe."})
                return
            account = self.manager.upsert_account(user_id, label=label or None, metadata=metadata, enabled=enabled)
            exchange_payload = payload.get("exchange") or {}
            if exchange_payload:
                try:
                    cred = _build_credential(exchange_payload, user_id=user_id, env_path=DEFAULT_ENV_PATH)
                    self.manager.upsert_exchange(user_id, cred)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
            _save_with_backup(self.manager, self.accounts_path)
            tmp_manager = AccountManager([account])
            self._send_json(201, _serialize(tmp_manager, self.accounts_path))
            return

        self.send_error(404, "Ruta no encontrada")

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if parts[:2] != ["api", "accounts"]:
            self.send_error(404, "Ruta no encontrada")
            return
        payload = self._read_json()
        if payload is None:
            return

        if len(parts) == 3:
            # Actualiza cuenta (label/enabled/metadata)
            user_id = parts[2]
            try:
                self.manager.get_account(user_id)
            except KeyError:
                self._send_json(404, {"error": f"No existe la cuenta '{user_id}'."})
                return
            new_id = (payload.get("id") or payload.get("user_id") or user_id).strip()
            if not new_id:
                self._send_json(400, {"error": "id no puede ser vacío."})
                return
            if new_id != user_id and new_id in {a.user_id for a in self.manager.list_accounts()}:
                self._send_json(409, {"error": f"La cuenta '{new_id}' ya existe."})
                return

            label = payload.get("label")
            enabled = payload.get("enabled")
            metadata = payload.get("metadata")
            if new_id != user_id:
                self.manager.rename_account(user_id, new_id)
                user_id = new_id
            self.manager.upsert_account(
                user_id,
                label=label if label is not None else None,
                metadata=metadata if isinstance(metadata, dict) else None,
                enabled=enabled if enabled is not None else None,
            )
            _save_with_backup(self.manager, self.accounts_path)
            self._send_json(200, self._snapshot())
            return

        if len(parts) == 4 and parts[3] == "exchange":
            user_id = parts[2]
            try:
                account = self.manager.get_account(user_id)
            except KeyError:
                self._send_json(404, {"error": f"No existe la cuenta '{user_id}'."})
                return
            try:
                cred = _build_credential(payload, default_name=payload.get("exchange") or payload.get("name"), user_id=user_id, env_path=DEFAULT_ENV_PATH)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            # Mantener un único exchange por usuario: se limpia y se inserta el nuevo.
            account.exchanges = {}
            self.manager.upsert_exchange(user_id, cred)
            _save_with_backup(self.manager, self.accounts_path)
            self._send_json(200, self._snapshot())
            return

        self.send_error(404, "Ruta no encontrada")

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if parts[:2] != ["api", "accounts"] or len(parts) != 3:
            self.send_error(404, "Ruta no encontrada")
            return
        user_id = parts[2]
        try:
            account = self.manager.get_account(user_id)
        except KeyError:
            self._send_json(404, {"error": f"No existe la cuenta '{user_id}'."})
            return
        # Borrado lógico: enabled = False
        account.enabled = False
        _save_with_backup(self.manager, self.accounts_path)
        self._send_json(200, self._snapshot())


def _build_handler(manager: AccountManager, accounts_path: Path, html_path: Path):
    class _Handler(DashCRUDHandler):
        pass

    _Handler.manager = manager
    _Handler.accounts_path = accounts_path
    _Handler.html_path = html_path
    return _Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="DashCRUD: server HTTP para cuentas/exchanges.")
    parser.add_argument("--accounts", type=str, default=DEFAULT_ACCOUNTS_PATH, help="Archivo YAML/JSON de cuentas.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host de escucha (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8050, help="Puerto de escucha (default 8050).")
    parser.add_argument("--html", type=str, default=None, help="Ruta del HTML del dashboard.")
    args = parser.parse_args()

    accounts_path = Path(args.accounts)
    html_path = Path(args.html) if args.html else DEFAULT_HTML
    manager = _load_manager(accounts_path)
    handler = _build_handler(manager, accounts_path, html_path)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[INFO] DashCRUD en http://{args.host}:{args.port}")
    print(f"[INFO] Archivo de cuentas: {accounts_path}")
    print(f"[INFO] HTML: {html_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Detenido por el usuario.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
