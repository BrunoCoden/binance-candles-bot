# telegram_bot_commands.py
"""
Bot sencillo que atiende comandos de Telegram relacionados con la estrategia.
Actualmente soporta:
    /estavivo  -> devuelve el mismo estado que produce el heartbeat.
    /dash      -> devuelve la URL del DashCRUD.
    /usuarios  -> lista usuarios activos y sus exchanges.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, Optional

import requests
from dotenv import load_dotenv

from heartbeat_monitor import generate_systemd_heartbeat_message, required_services_from_env
from trading.accounts.manager import AccountManager


def _parse_chat_ids(chat_ids_env: str | None) -> list[str]:
    if not chat_ids_env:
        return []
    parts = [part.strip() for part in chat_ids_env.replace(";", ",").split(",")]
    return [part for part in parts if part]


def _send_message(token: str, chat_id: int | str, text: str, reply_to: Optional[int] = None) -> None:
    payload = {
        "chat_id": chat_id,
        "text": text,
    }
    if reply_to is not None:
        payload["reply_to_message_id"] = reply_to

    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
            timeout=10,
        ).raise_for_status()
    except Exception as exc:
        print(f"[BOT][WARN] No se pudo enviar respuesta a Telegram ({chat_id}): {exc}")


def _fetch_updates(token: str, offset: Optional[int]) -> dict:
    params = {
        "timeout": 30,
    }
    if offset is not None:
        params["offset"] = offset
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params=params,
            timeout=35,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[BOT][WARN] Error consultando getUpdates: {exc}")
        return {"ok": False, "result": []}


def _is_authorized(chat_id: int | str, allowed: Iterable[str]) -> bool:
    if not allowed:
        return True
    return str(chat_id) in allowed


def _normalize_command(text: str) -> str:
    if not text:
        return ""
    return text.strip().lower()

def _extract_command_and_arg(raw_text: str | None) -> tuple[str, str]:
    if not raw_text:
        return "", ""
    text = raw_text.strip()
    if not text.startswith("/"):
        return "", ""
    parts = text.split(maxsplit=1)
    cmd = parts[0].strip().lower()
    arg = parts[1].strip() if len(parts) == 2 else ""
    return cmd, arg


def _save_accounts_with_backup(manager: AccountManager, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ts = int(time.time())
        backup = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            backup.write_bytes(path.read_bytes())
        except Exception:
            pass
    manager.save_to_file(path)


def _handle_command(
    *,
    token: str,
    chat_id: int,
    message_id: Optional[int],
    command: str,
    arg: str,
    required_services: list[str],
) -> None:
    if command.startswith("/estavivo"):
        report = generate_systemd_heartbeat_message(required_services)
        _send_message(token, chat_id, report, reply_to=message_id)
        return

    if command.startswith("/dash"):
        url = os.getenv("DASHCRUD_PUBLIC_URL", "https://167.126.0.127")
        _send_message(token, chat_id, url, reply_to=message_id)
        return

    if command.startswith("/usuarios"):
        accounts_path = os.getenv("WATCHER_ACCOUNTS_FILE", "trading/accounts/oci_accounts.yaml")
        try:
            manager = AccountManager.from_file(Path(accounts_path))
        except Exception as exc:
            _send_message(token, chat_id, f"No pude leer cuentas ({accounts_path}): {exc}", reply_to=message_id)
            return
        lines = ["Usuarios:"]
        for account in manager.list_accounts():
            exchanges = sorted((account.exchanges or {}).keys())
            if not exchanges:
                continue
            state = "ON" if account.enabled else "OFF"
            lines.append(f"- {account.user_id} [{state}]: {', '.join(exchanges)}")
        if len(lines) == 1:
            lines.append("(ninguno)")
        _send_message(token, chat_id, "\n".join(lines), reply_to=message_id)
        return

    if command in {"/habilitar", "/deshabilitar"}:
        user_id = arg.strip()
        if not user_id:
            _send_message(token, chat_id, f"Uso: {command} <user_id>", reply_to=message_id)
            return
        accounts_path = Path(os.getenv("WATCHER_ACCOUNTS_FILE", "trading/accounts/oci_accounts.yaml"))
        try:
            manager = AccountManager.from_file(accounts_path)
            account = manager.get_account(user_id)
        except Exception as exc:
            _send_message(token, chat_id, f"No pude cargar '{user_id}' ({accounts_path}): {exc}", reply_to=message_id)
            return

        desired = command == "/habilitar"
        if account.enabled == desired:
            _send_message(token, chat_id, f"{user_id} ya está {'habilitado' if desired else 'deshabilitado'}.", reply_to=message_id)
            return

        account.enabled = desired
        try:
            _save_accounts_with_backup(manager, accounts_path)
        except Exception as exc:
            _send_message(token, chat_id, f"No pude guardar cambios en {accounts_path}: {exc}", reply_to=message_id)
            return

        note = ""
        if not desired:
            note = "\nSe intentará cerrar posiciones (según lógica del watcher) cuando recargue cuentas."
        _send_message(token, chat_id, f"{user_id} {'habilitado' if desired else 'deshabilitado'} ✅{note}", reply_to=message_id)
        return

    if command in {"/start", "/help"}:
        help_text = (
            "Comandos disponibles:\n"
            "• /estavivo — chequea los procesos críticos y devuelve el estado actual.\n"
            "• /dash — devuelve la URL del DashCRUD.\n"
            "• /usuarios — lista usuarios activos y sus exchanges.\n"
            "• /habilitar <user_id> — habilita el usuario.\n"
            "• /deshabilitar <user_id> — deshabilita el usuario (y el watcher intentará cerrar posiciones).\n"
            "Los mensajes siguen el formato del heartbeat automático."
        )
        _send_message(token, chat_id, help_text, reply_to=message_id)
        return


def main() -> None:
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN no configurado.")

    allowed_chat_ids = _parse_chat_ids(os.getenv("TELEGRAM_CHAT_IDS"))
    required_services = required_services_from_env(None)
    if not required_services:
        raise SystemExit("HEARTBEAT_SERVICES vacío; definí servicios a monitorear.")

    print("[BOT] Telegram command listener iniciado.")
    offset: Optional[int] = None

    while True:
        data = _fetch_updates(token, offset)
        if not data.get("ok"):
            time.sleep(5)
            continue

        for update in data.get("result", []):
            offset = update["update_id"] + 1

            message = update.get("message") or update.get("channel_post")
            if not message:
                continue
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            if chat_id is None:
                continue
            if not _is_authorized(chat_id, allowed_chat_ids):
                continue

            text = message.get("text")
            command, arg = _extract_command_and_arg(text)
            if not command:
                continue

            _handle_command(
                token=token,
                chat_id=chat_id,
                message_id=message.get("message_id"),
                command=command,
                arg=arg,
                required_services=required_services,
            )

        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[BOT] Finalizado por el usuario.")
