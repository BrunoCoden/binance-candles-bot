#!/usr/bin/env python3
"""
DashCRUD: dashboard mínimo para CRUD de cuentas/exchanges.

- Usa YAML/JSON en trading/accounts/* para persistir.
- No expone secretos; solo nombres de variables de entorno.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

# Asegura imports relativos al repo
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading.accounts.manager import AccountManager
from trading.accounts.models import AccountConfig, ExchangeCredential, ExchangeEnvironment

DEFAULT_ACCOUNTS_PATH = Path("trading/accounts/oci_accounts.yaml")
DEFAULT_HTML = REPO_ROOT / "trading/accounts/dashcrud.html"


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


def _build_credential(payload: Dict[str, Any], default_name: str | None = None) -> ExchangeCredential:
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
    if not api_key_env or not api_secret_env:
        raise ValueError("api_key_env y api_secret_env son obligatorios.")

    notional_val = payload.get("notional_usdt")
    leverage_val = payload.get("leverage")
    notional = float(notional_val) if notional_val not in (None, "", False) else None
    leverage = int(leverage_val) if leverage_val not in (None, "", False) else None
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}

    return ExchangeCredential(
        exchange=name,
        api_key_env=api_key_env,
        api_secret_env=api_secret_env,
        environment=environment,
        notional_usdt=notional,
        leverage=leverage,
        extra=extra,
    )


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
            # Exchange es obligatorio para crear
            exchange_payload = payload.get("exchange") or {}
            try:
                cred = _build_credential(exchange_payload)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            account = self.manager.upsert_account(user_id, label=label or None, metadata=metadata, enabled=enabled)
            self.manager.upsert_exchange(user_id, cred)
            self.manager.save_to_file(self.accounts_path)
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
            self.manager.save_to_file(self.accounts_path)
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
                cred = _build_credential(payload, default_name=payload.get("exchange") or payload.get("name"))
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            # Mantener un único exchange por usuario: se limpia y se inserta el nuevo.
            account.exchanges = {}
            self.manager.upsert_exchange(user_id, cred)
            self.manager.save_to_file(self.accounts_path)
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
        self.manager.save_to_file(self.accounts_path)
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
