# telegram_bot_commands.py
"""
Bot sencillo que atiende comandos de Telegram relacionados con la estrategia.
Actualmente soporta:
    /estavivo  -> devuelve el mismo estado que produce el heartbeat.
    /usuarios  -> lista usuarios activos y sus exchanges.
    /notional  -> actualiza el notional por usuario/exchange.
"""
from __future__ import annotations

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import requests
from dotenv import load_dotenv

from heartbeat_monitor import generate_systemd_heartbeat_message, required_services_from_env
from trading.accounts.manager import AccountManager
from trading.accounts.models import ExchangeCredential, ExchangeEnvironment

_PENDING_ENV_UPDATES: dict[str, dict[str, object]] = {}


def _load_thresholds(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _binance_position_details(cred: ExchangeCredential, symbol: str) -> tuple[float | None, float | None]:
    try:
        from binance.um_futures import UMFutures
    except Exception as exc:
        return None, None
    try:
        api_key, api_secret = cred.resolve_keys(os.environ)
        client = UMFutures(key=api_key, secret=api_secret)
        positions = client.get_position_risk(symbol=symbol)
        if not positions:
            return 0.0, None
        pos = positions[0]
        amt = float(pos.get("positionAmt") or 0.0)
        entry = float(pos.get("entryPrice") or 0.0)
        return amt, (entry if entry > 0 else None)
    except Exception:
        return None, None


def _bybit_position_details(cred: ExchangeCredential, symbol: str) -> tuple[float | None, float | None]:
    try:
        from pybit.unified_trading import HTTP
    except Exception:
        return None, None
    try:
        api_key, api_secret = cred.resolve_keys(os.environ)
        is_testnet = cred.environment != ExchangeEnvironment.LIVE
        session = HTTP(testnet=is_testnet, api_key=api_key, api_secret=api_secret)
        resp = session.get_positions(category="linear", symbol=symbol)
        data = resp.get("result", {}).get("list", []) if isinstance(resp, dict) else []
        if not data:
            return 0.0, None
        pos = data[0]
        size = float(pos.get("size") or 0.0)
        side = (pos.get("side") or "").lower()
        if side == "sell":
            size = -abs(size)
        elif side == "buy":
            size = abs(size)
        entry = float(pos.get("avgPrice") or 0.0)
        return size, (entry if entry > 0 else None)
    except Exception:
        return None, None

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

def _load_env_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _write_env_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    path.write_text(text, encoding="utf-8")


def _update_env_vars(path: Path, updates: dict[str, str]) -> None:
    """
    Actualiza/crea variables en el archivo .env sin tocar otras entradas.
    """
    existing = _load_env_lines(path)
    new_lines: list[str] = []
    remaining = dict(updates)
    for line in existing:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key, _ = line.split("=", 1)
        key = key.strip()
        if key in remaining:
            new_lines.append(f"{key}={remaining.pop(key)}")
        else:
            new_lines.append(line)
    for key, value in remaining.items():
        new_lines.append(f"{key}={value}")
    _write_env_lines(path, new_lines)


def _parse_kv_pairs(raw: str) -> tuple[list[tuple[str, str]], list[str]]:
    pairs: list[tuple[str, str]] = []
    errors: list[str] = []
    if not raw:
        return pairs, errors
    tokens = raw.split()
    for token in tokens:
        if "=" not in token:
            errors.append(token)
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            errors.append(token)
            continue
        if " " in key or " " in value:
            errors.append(token)
            continue
        pairs.append((key, value))
    return pairs, errors


def _parse_args_map(raw: str) -> tuple[dict[str, str], list[str]]:
    pairs, errors = _parse_kv_pairs(raw)
    data: dict[str, str] = {}
    for key, value in pairs:
        data[key] = value
    return data, errors


def _parse_notional_args(raw: str) -> tuple[str, str, float, str | None]:
    if not raw or not raw.strip():
        return "", "", 0.0, "Uso: /notional <user_id> <exchange> <monto_usdt>"
    if "=" in raw:
        args_map, errors = _parse_args_map(raw)
        if errors:
            return "", "", 0.0, "Formato inválido. Usá KEY=VALUE sin espacios."
        user_id = args_map.get("user_id") or args_map.get("user")
        exchange = args_map.get("exchange") or args_map.get("ex")
        amount = (
            args_map.get("notional_usdt")
            or args_map.get("notional")
            or args_map.get("amount")
        )
    else:
        parts = raw.split()
        if len(parts) != 3:
            return "", "", 0.0, "Uso: /notional <user_id> <exchange> <monto_usdt>"
        user_id, exchange, amount = parts

    if not user_id or not exchange or not amount:
        return "", "", 0.0, "Uso: /notional <user_id> <exchange> <monto_usdt>"
    try:
        notional = float(amount)
    except Exception:
        return "", "", 0.0, "Monto inválido. Ejemplo: /notional diego binance 30"
    if notional <= 0:
        return "", "", 0.0, "Monto inválido. Debe ser mayor a 0."
    return str(user_id), str(exchange).lower(), notional, None


def _parse_single_kv(raw: str) -> tuple[str | None, str | None]:
    text = (raw or "").strip()
    if not text or "=" not in text:
        return None, None
    key, value = text.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value or " " in key or " " in value:
        return None, None
    return key, value


def _begin_interactive_env(chat_id: int | str, user_id: str) -> None:
    _PENDING_ENV_UPDATES[str(chat_id)] = {"mode": "env", "user_id": user_id, "vars": {}}


def _begin_interactive_user(chat_id: int | str) -> None:
    _PENDING_ENV_UPDATES[str(chat_id)] = {"mode": "user", "fields": {}}


def _begin_interactive(chat_id: int | str, user_id: str) -> None:
    _begin_interactive_env(chat_id, user_id)


def _pending_state(chat_id: int | str) -> dict[str, object] | None:
    return _PENDING_ENV_UPDATES.get(str(chat_id))


def _clear_pending(chat_id: int | str) -> None:
    _PENDING_ENV_UPDATES.pop(str(chat_id), None)


def _apply_pending_env_updates(
    *,
    token: str,
    chat_id: int,
    message_id: Optional[int],
    env_path: Path,
) -> None:
    state = _pending_state(chat_id)
    if not state or state.get("mode") != "env":
        _send_message(token, chat_id, "No hay cambios pendientes para aplicar.", reply_to=message_id)
        return
    vars_map = state.get("vars") or {}
    if not isinstance(vars_map, dict) or not vars_map:
        _send_message(token, chat_id, "No hay variables pendientes para aplicar.", reply_to=message_id)
        return
    updates = {str(k): str(v) for k, v in vars_map.items()}
    try:
        _update_env_vars(env_path, updates)
    except Exception as exc:
        _send_message(token, chat_id, f"No pude actualizar {env_path}: {exc}", reply_to=message_id)
        return
    applied = ", ".join(sorted(updates.keys()))
    _clear_pending(chat_id)
    _send_message(
        token,
        chat_id,
        f"Variables aplicadas: {applied}.\nReiniciando watcher...",
        reply_to=message_id,
    )
    service = os.getenv("WATCHER_SERVICE_NAME", "bot-watcher.service")
    ok, detail = _restart_service(service)
    if ok:
        _send_message(token, chat_id, f"Watcher reiniciado: {service}", reply_to=message_id)
    else:
        _send_message(token, chat_id, f"No pude reiniciar {service}: {detail}", reply_to=message_id)


def _apply_pending_user_create(
    *,
    token: str,
    chat_id: int,
    message_id: Optional[int],
    accounts_path: Path,
) -> None:
    state = _pending_state(chat_id)
    if not state or state.get("mode") != "user":
        _send_message(token, chat_id, "No hay alta de usuario pendiente.", reply_to=message_id)
        return
    fields = state.get("fields") or {}
    if not isinstance(fields, dict):
        _send_message(token, chat_id, "No hay datos para el alta de usuario.", reply_to=message_id)
        return
    required = ["user_id", "exchange", "api_key_env", "api_secret_env", "environment", "notional_usdt", "symbol"]
    missing = [key for key in required if not fields.get(key)]
    if missing:
        _send_message(
            token,
            chat_id,
            f"Faltan campos requeridos: {', '.join(missing)}",
            reply_to=message_id,
        )
        return
    user_id = str(fields["user_id"])
    exchange = str(fields["exchange"]).lower()
    try:
        environment = ExchangeEnvironment(str(fields["environment"]).lower())
    except Exception:
        _send_message(token, chat_id, "environment inválido (usar live/testnet).", reply_to=message_id)
        return
    try:
        notional = float(fields["notional_usdt"])
    except Exception:
        _send_message(token, chat_id, "notional_usdt inválido.", reply_to=message_id)
        return
    leverage = None
    if fields.get("leverage"):
        try:
            leverage = int(fields["leverage"])
        except Exception:
            _send_message(token, chat_id, "leverage inválido.", reply_to=message_id)
            return
    enabled = str(fields.get("enabled", "true")).lower() != "false"
    label = str(fields.get("label", user_id))
    extra: dict[str, str] = {}
    for key, value in fields.items():
        if str(key).startswith("extra_"):
            extra[str(key).removeprefix("extra_")] = str(value)
    if "symbol" not in extra:
        extra["symbol"] = str(fields["symbol"])
    if "margin_mode" not in extra:
        extra["margin_mode"] = "isolated"

    try:
        if accounts_path.exists():
            manager = AccountManager.from_file(accounts_path)
        else:
            manager = AccountManager.empty()
        account = manager.upsert_account(user_id, label=label, enabled=enabled)
        credential = ExchangeCredential(
            exchange=exchange,
            api_key_env=str(fields["api_key_env"]),
            api_secret_env=str(fields["api_secret_env"]),
            environment=environment,
            notional_usdt=notional,
            leverage=leverage if leverage is not None else 5,
            extra=extra,
        )
        manager.upsert_exchange(account.user_id, credential)
        _save_accounts_with_backup(manager, accounts_path)
    except Exception as exc:
        _send_message(token, chat_id, f"No pude guardar usuario ({accounts_path}): {exc}", reply_to=message_id)
        return
    _clear_pending(chat_id)
    ok, detail = _restart_service(os.getenv("WATCHER_SERVICE_NAME", "bot-watcher.service"))
    if ok:
        _send_message(token, chat_id, f"Usuario {user_id} creado/actualizado ✅", reply_to=message_id)
    else:
        _send_message(token, chat_id, f"Usuario {user_id} creado, pero no pude reiniciar watcher: {detail}", reply_to=message_id)


def _restart_service(service: str) -> tuple[bool, str]:
    try:
        subprocess.run(
            ["sudo", "-n", "systemctl", "restart", service],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, ""
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else exc.stdout.strip()
        return False, detail


def _ensure_chat_allowed(
    *,
    token: str,
    chat_id: int | str,
    allowed: list[str],
    env_path: Path,
) -> bool:
    chat_id_str = str(chat_id)
    if chat_id_str in allowed:
        return True

    # Auto-agrega el chat al allowlist y al .env para que reciba alertas.
    updated = [cid for cid in allowed if cid]
    if chat_id_str not in updated:
        updated.append(chat_id_str)
    try:
        _update_env_vars(env_path, {"TELEGRAM_CHAT_IDS": ",".join(updated)})
        allowed[:] = updated
        _send_message(
            token,
            chat_id,
            "Chat habilitado automaticamente. Aplicando cambios...",
        )
    except Exception as exc:
        _send_message(token, chat_id, f"No pude habilitar el chat: {exc}")
        return False

    service = os.getenv("WATCHER_SERVICE_NAME", "bot-watcher.service")
    ok, detail = _restart_service(service)
    if ok:
        _send_message(token, chat_id, f"Watcher reiniciado: {service}")
    else:
        _send_message(token, chat_id, f"No pude reiniciar {service}: {detail}")
    return True


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

    if command.startswith("/posicion"):
        accounts_path = Path(os.getenv("WATCHER_ACCOUNTS_FILE", "trading/accounts/oci_accounts.yaml"))
        thresholds_path = Path(os.getenv("WATCHER_THRESHOLDS_FILE", "backtest/backtestTR/pending_thresholds.json"))
        try:
            manager = AccountManager.from_file(accounts_path)
        except Exception as exc:
            _send_message(token, chat_id, f"No pude cargar cuentas: {exc}", reply_to=message_id)
            return

        arg_map, errors = _parse_args_map(arg)
        user_filter = ""
        exchange_filter = ""
        if not errors and arg_map:
            user_filter = str(arg_map.get("user_id", "")).strip()
            exchange_filter = str(arg_map.get("exchange", "")).strip().lower()
        else:
            tokens = arg.split()
            if tokens:
                if len(tokens) >= 2 and tokens[-1].lower() in {"binance", "bybit", "dydx"}:
                    exchange_filter = tokens[-1].lower()
                    user_filter = " ".join(tokens[:-1]).strip()
                else:
                    user_filter = " ".join(tokens).strip()

        thresholds = _load_thresholds(thresholds_path)
        rows = []
        for account in manager.list_accounts():
            if not account.enabled:
                continue
            if user_filter and account.user_id != user_filter:
                continue
            for ex_name, cred in (account.exchanges or {}).items():
                if exchange_filter and ex_name.lower() != exchange_filter:
                    continue
                if isinstance(cred.extra, dict) and cred.extra.get("enabled") is False:
                    continue
                symbol = (cred.extra or {}).get("symbol") or "ETHUSDT"
                pos_amt = None
                entry_price = None
                if ex_name.lower() == "binance":
                    pos_amt, entry_price = _binance_position_details(cred, symbol)
                elif ex_name.lower() == "bybit":
                    pos_amt, entry_price = _bybit_position_details(cred, symbol)
                else:
                    continue
                if pos_amt is None:
                    status = "pos=ERROR"
                else:
                    status = f"pos={pos_amt:.4f}"
                if entry_price:
                    status += f" entry={entry_price:.4f}"
                th = None
                for t in thresholds:
                    if t.get("user_id") == account.user_id and t.get("exchange") == ex_name and t.get("symbol") == symbol:
                        th = t
                        break
                if th:
                    status += f" SL={float(th.get('loss_price') or 0):.4f}"
                    gain = th.get('gain_price')
                    if gain not in (None, ""):
                        status += f" TP={float(gain):.4f}"
                rows.append(f"- {account.user_id}/{ex_name} {symbol} → {status}")

        if not rows:
            _send_message(token, chat_id, "No hay posiciones abiertas (o no se pudo consultar).", reply_to=message_id)
        else:
            _send_message(token, chat_id, "Posiciones:\n" + "\n".join(rows), reply_to=message_id)
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
            exchanges_data = account.exchanges or {}
            if not exchanges_data:
                continue
            state = "ON" if account.enabled else "OFF"
            details = []
            for ex_name in sorted(exchanges_data.keys()):
                cred = exchanges_data.get(ex_name)
                if not cred:
                    continue
                notional = cred.notional_usdt
                notional_label = f"notional={notional:.2f}" if isinstance(notional, (int, float)) else "notional=NA"
                leverage = cred.leverage
                leverage_label = f"lev={leverage}x" if isinstance(leverage, int) and leverage > 0 else "lev=NA"
                margin_mode = None
                if isinstance(cred.extra, dict):
                    margin_mode = cred.extra.get("margin_mode")
                margin_label = f"margin={margin_mode}" if margin_mode else "margin=NA"
                details.append(f"{ex_name} ({notional_label} {leverage_label} {margin_label})")
            if not details:
                continue
            lines.append(f"- {account.user_id} [{state}]: {', '.join(details)}")
        if len(lines) == 1:
            lines.append("(ninguno)")
        _send_message(token, chat_id, "\n".join(lines), reply_to=message_id)
        return

    if command.startswith("/notional"):
        user_id, exchange, notional, error = _parse_notional_args(arg)
        if error:
            _send_message(token, chat_id, error, reply_to=message_id)
            return
        accounts_path = Path(os.getenv("WATCHER_ACCOUNTS_FILE", "trading/accounts/oci_accounts.yaml"))
        try:
            manager = AccountManager.from_file(accounts_path)
            cred = manager.get_exchange_credential(user_id, exchange)
        except Exception as exc:
            _send_message(token, chat_id, f"No pude cargar {user_id}/{exchange}: {exc}", reply_to=message_id)
            return
        cred.notional_usdt = notional
        try:
            manager.upsert_exchange(user_id, cred)
            _save_accounts_with_backup(manager, accounts_path)
        except Exception as exc:
            _send_message(token, chat_id, f"No pude guardar notional en {accounts_path}: {exc}", reply_to=message_id)
            return
        ok, detail = _restart_service(os.getenv("WATCHER_SERVICE_NAME", "bot-watcher.service"))
        if ok:
            _send_message(
                token,
                chat_id,
                f"Notional actualizado: {user_id}/{exchange} -> {notional:.2f}.\nWatcher reiniciado ✅",
                reply_to=message_id,
            )
        else:
            _send_message(
                token,
                chat_id,
                f"Notional actualizado: {user_id}/{exchange} -> {notional:.2f}.\n"
                f"No pude reiniciar watcher: {detail}",
                reply_to=message_id,
            )
        return

    if command in {"/alta_usuario", "/crear_usuario"}:
        accounts_path = Path(os.getenv("WATCHER_ACCOUNTS_FILE", "trading/accounts/oci_accounts.yaml"))
        if not arg.strip():
            _begin_interactive_user(chat_id)
            _send_message(
                token,
                chat_id,
                "Modo interactivo de alta activo. Envia KEY=VALUE uno por uno.\n"
                "Requeridos: user_id, exchange, api_key_env, api_secret_env, environment, notional_usdt, symbol.\n"
                "Opcionales: leverage, label, enabled, extra_* (ej: extra_margin_mode=isolated).\n"
                "Cuando termines, usa /aplicar.",
                reply_to=message_id,
            )
            return
        args_map, errors = _parse_args_map(arg)
        if errors:
            _send_message(
                token,
                chat_id,
                "Formato invalido. Usa KEY=VALUE separados por espacio (sin espacios dentro).",
                reply_to=message_id,
            )
            return
        _PENDING_ENV_UPDATES[str(chat_id)] = {"mode": "user", "fields": args_map}
        _apply_pending_user_create(token=token, chat_id=chat_id, message_id=message_id, accounts_path=accounts_path)
        return

    if command in {"/habilitar", "/deshabilitar"}:
        args_map, errors = _parse_args_map(arg)
        user_id = ""
        exchange_name = ""
        extra_raw = ""

        if not errors and args_map:
            user_id = str(args_map.get("user_id", "")).strip()
            exchange_name = str(args_map.get("exchange", "")).strip().lower()
            extra_raw = ""
        else:
            tokens = arg.split()
            if tokens:
                if len(tokens) >= 2 and tokens[-1].lower() in {"binance", "bybit", "dydx"}:
                    exchange_name = tokens[-1].lower()
                    user_id = " ".join(tokens[:-1]).strip()
                else:
                    user_id = " ".join(tokens).strip()
            extra_raw = ""

        if not user_id:
            _send_message(token, chat_id, f"Uso: {command} <user_id> [exchange]", reply_to=message_id)
            return
        accounts_path = Path(os.getenv("WATCHER_ACCOUNTS_FILE", "trading/accounts/oci_accounts.yaml"))
        env_path = Path(os.getenv("WATCHER_ENV_FILE", ".env"))
        try:
            manager = AccountManager.from_file(accounts_path)
            account = manager.get_account(user_id)
        except Exception as exc:
            _send_message(token, chat_id, f"No pude cargar '{user_id}' ({accounts_path}): {exc}", reply_to=message_id)
            return

        desired = command == "/habilitar"

        if exchange_name:
            if exchange_name not in account.exchanges:
                _send_message(token, chat_id, f"{user_id} no tiene exchange '{exchange_name}'.", reply_to=message_id)
                return
            ex = account.exchanges[exchange_name]
            current = bool(ex.extra.get("enabled", True))
            if current == desired:
                _send_message(token, chat_id, f"{user_id}/{exchange_name} ya esta {'habilitado' if desired else 'deshabilitado'}.", reply_to=message_id)
                return
            ex.extra["enabled"] = desired
        else:
            if account.enabled == desired:
                _send_message(token, chat_id, f"{user_id} ya esta {'habilitado' if desired else 'deshabilitado'}.", reply_to=message_id)
                return
            account.enabled = desired

        try:
            _save_accounts_with_backup(manager, accounts_path)
        except Exception as exc:
            _send_message(token, chat_id, f"No pude guardar cambios en {accounts_path}: {exc}", reply_to=message_id)
            return

        note = ""
        env_note = ""
        applied_keys: list[str] = []
        if desired and extra_raw:
            pairs, errors = _parse_kv_pairs(extra_raw)
            if errors:
                _send_message(
                    token,
                    chat_id,
                    "Formato invalido para variables. Usa KEY=VALUE separados por espacio (sin espacios dentro).",
                    reply_to=message_id,
                )
                return
            updates = {k: v for k, v in pairs}
            try:
                _update_env_vars(env_path, updates)
                applied_keys = sorted(updates.keys())
                env_note = f"\nVariables aplicadas: {', '.join(applied_keys)}."
            except Exception as exc:
                _send_message(token, chat_id, f"No pude actualizar {env_path}: {exc}", reply_to=message_id)
                return
        if not desired:
            note = "\nSe intentara cerrar posiciones (segun logica del watcher) cuando recargue cuentas."
        if desired and env_note:
            note = f"{note}{env_note}\nReinicia servicios del bot para tomar el .env nuevo."
        if desired and not extra_raw:
            _begin_interactive(chat_id, user_id)
            note = (
                f"{note}\nModo interactivo activo: envia KEY=VALUE uno por uno.\n"
                "Luego usa /aplicar para guardar en .env o /cancelar para descartar."
            )
        target = f"{user_id}/{exchange_name}" if exchange_name else user_id
        _send_message(token, chat_id, f"{target} {'habilitado' if desired else 'deshabilitado'} ✅{note}", reply_to=message_id)
        return
    if command == "/aplicar":
        env_path = Path(os.getenv("WATCHER_ENV_FILE", ".env"))
        accounts_path = Path(os.getenv("WATCHER_ACCOUNTS_FILE", "trading/accounts/oci_accounts.yaml"))
        state = _pending_state(chat_id)
        if state and state.get("mode") == "user":
            _apply_pending_user_create(token=token, chat_id=chat_id, message_id=message_id, accounts_path=accounts_path)
        else:
            _apply_pending_env_updates(token=token, chat_id=chat_id, message_id=message_id, env_path=env_path)
        return

    if command == "/cancelar":
        if _pending_state(chat_id):
            _clear_pending(chat_id)
            _send_message(token, chat_id, "Cambios pendientes descartados.", reply_to=message_id)
        else:
            _send_message(token, chat_id, "No hay cambios pendientes para cancelar.", reply_to=message_id)
        return

    if command in {"/reiniciar_watcher", "/restart_watcher"}:
        service = os.getenv("WATCHER_SERVICE_NAME", "bot-watcher.service")
        ok, detail = _restart_service(service)
        if ok:
            _send_message(token, chat_id, f"Watcher reiniciado: {service}", reply_to=message_id)
        else:
            _send_message(token, chat_id, f"No pude reiniciar {service}: {detail}", reply_to=message_id)
        return

    if command in {"/start", "/help"}:
        help_text = (
            "Comandos disponibles:\n"
            "• /estavivo — chequea los procesos criticos y devuelve el estado actual.\n"
            "• /usuarios — lista usuarios activos y sus exchanges.\n"
            "• /notional — actualiza el notional de un usuario/exchange.\n"
            "    Uso: /notional <user_id> <exchange> <monto_usdt>\n"
            "    Tambien acepta: /notional user_id=... exchange=... notional_usdt=...\n"
            "• /alta_usuario — modo interactivo para alta de usuarios (usar KEY=VALUE y luego /aplicar).\n"
            "    Tambien acepta formato inline: user_id=... exchange=... api_key_env=... api_secret_env=... environment=live|testnet notional_usdt=... symbol=...\n"
            "    Opcional: leverage=... label=... enabled=true|false extra_* (ej: extra_margin_mode=isolated).\n"
            "• /habilitar <user_id> — habilita el usuario en /home/ubuntu/bot/trading/accounts/oci_accounts.yaml.\n"
            "    Opcional: agregar variables KEY=VALUE para dejar credenciales listas (se guardan en /home/ubuntu/bot/.env).\n"
            "• /deshabilitar <user_id> — deshabilita el usuario (y el watcher intentara cerrar posiciones).\n"
            "• /aplicar — aplica variables pendientes del modo interactivo.\n"
            "• /cancelar — cancela el modo interactivo.\n"
            "• /reiniciar_watcher — reinicia el servicio del watcher.\n"
            "Los mensajes siguen el formato del heartbeat automatico."
        )
        _send_message(token, chat_id, help_text, reply_to=message_id)
        return


def main() -> None:
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN no configurado.")

    allowed_chat_ids = _parse_chat_ids(os.getenv("TELEGRAM_CHAT_IDS"))
    env_path = Path(os.getenv("WATCHER_ENV_FILE", ".env"))
    required_services = required_services_from_env(None)
    if not required_services:
        raise SystemExit("HEARTBEAT_SERVICES vacio; defini servicios a monitorear.")

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
                if not _ensure_chat_allowed(
                    token=token,
                    chat_id=chat_id,
                    allowed=allowed_chat_ids,
                    env_path=env_path,
                ):
                    continue

            text = message.get("text")
            command, arg = _extract_command_and_arg(text)
            if not command:
                pending = _pending_state(chat_id)
                if pending and text:
                    key, value = _parse_single_kv(text)
                    if not key:
                        _send_message(
                            token,
                            chat_id,
                            "Formato invalido. Envia KEY=VALUE sin espacios. Usa /aplicar o /cancelar.",
                            reply_to=message.get("message_id"),
                        )
                        continue
                    if pending.get("mode") == "user":
                        fields = pending.get("fields")
                        if not isinstance(fields, dict):
                            fields = {}
                            pending["fields"] = fields
                        fields[key] = value
                    else:
                        vars_map = pending.get("vars")
                        if not isinstance(vars_map, dict):
                            vars_map = {}
                            pending["vars"] = vars_map
                        vars_map[key] = value
                    if pending.get("mode") == "user":
                        required = {"user_id", "exchange", "api_key_env", "api_secret_env", "environment", "notional_usdt", "symbol"}
                        fields = pending.get("fields") if isinstance(pending.get("fields"), dict) else {}
                        missing = sorted([r for r in required if r not in fields or not fields.get(r)])
                        status = f"Faltan: {', '.join(missing)}" if missing else "Campos requeridos completos."
                        msg = f"Campo registrado: {key}. {status} Envia otro o /aplicar."
                    else:
                        msg = f"Variable registrada: {key}. Envia otra o /aplicar."
                    _send_message(
                        token,
                        chat_id,
                        msg,
                        reply_to=message.get("message_id"),
                    )
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