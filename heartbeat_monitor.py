# heartbeat_monitor.py
"""
Heartbeat que envía cada cierto intervalo el estado de los procesos críticos
al bot de Telegram configurado en el .env.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from trade_logger import send_trade_notification
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    from trade_logger import send_trade_notification  # type: ignore


DEFAULT_PROCESS_LIST = (
    "python watcher_alertas.py;"
    "python backtest/order_fill_listener.py;"
    "python estrategiaBollinger.py"
)


@dataclass
class ProcessStatus:
    label: str
    running: bool
    matches: list[str]


def _parse_required_processes(value: str | None) -> list[str]:
    if not value:
        value = DEFAULT_PROCESS_LIST
    parts = [part.strip() for part in value.replace(",", ";").split(";")]
    return [part for part in parts if part]


def required_processes_from_env(override: str | None = None) -> list[str]:
    """
    Devuelve la lista de procesos a monitorear usando la env HEARTBEAT_PROCESSES.
    """
    env_value = override if override is not None else os.getenv("HEARTBEAT_PROCESSES")
    return _parse_required_processes(env_value)


def _list_process_commands() -> Sequence[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,command"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"No se pudo obtener la lista de procesos ({exc})") from exc
    lines = result.stdout.splitlines()
    if not lines:
        return []
    return lines[1:]  # descarta encabezado


def _evaluate_processes(required: Iterable[str], processes: Sequence[str]) -> list[ProcessStatus]:
    statuses: list[ProcessStatus] = []
    for label in required:
        matches = [proc for proc in processes if label in proc]
        statuses.append(ProcessStatus(label=label, running=bool(matches), matches=matches))
    return statuses


def _build_message(
    *,
    statuses: Sequence[ProcessStatus],
    tz: ZoneInfo,
) -> str:
    now = datetime.now(tz)
    header = f"[HEARTBEAT] {now.isoformat(timespec='seconds')}"
    lines = [header, ""]
    overall = "OK" if all(status.running for status in statuses) else "ALERTA"
    lines.append(f"Estado general: {overall}")
    lines.append("")
    for status in statuses:
        state = "OK" if status.running else "FALTA"
        lines.append(f"- {state} :: {status.label}")
        if status.running and status.matches:
            first = status.matches[0].strip()
            lines.append(f"    {first}")
    return "\n".join(lines)


def _resolve_timezone() -> ZoneInfo:
    tz_name = os.getenv("TZ", "UTC")
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def run_heartbeat(
    *,
    required_processes: list[str],
    interval_hours: float,
    once: bool,
) -> None:
    tz = _resolve_timezone()
    sleep_seconds = max(1.0, interval_hours * 3600.0)

    while True:
        message = generate_heartbeat_message(required_processes, tz=tz)
        send_trade_notification(message)
        if once:
            break
        time.sleep(sleep_seconds)


def generate_heartbeat_message(
    required_processes: list[str],
    tz: ZoneInfo | None = None,
) -> str:
    """
    Construye el mensaje resumido de estado para los procesos indicados.
    """
    tz = tz or _resolve_timezone()
    processes = _list_process_commands()
    statuses = _evaluate_processes(required_processes, processes)
    return _build_message(statuses=statuses, tz=tz)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Heartbeat que avisa por Telegram si los procesos clave están activos."
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "12")),
        help="Intervalo entre notificaciones (en horas).",
    )
    parser.add_argument(
        "--processes",
        type=str,
        default=os.getenv("HEARTBEAT_PROCESSES"),
        help="Lista de procesos a monitorear (separador ';' o ',').",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Enviar solo una notificación y salir (útil para pruebas manuales).",
    )

    args = parser.parse_args()

    required_processes = required_processes_from_env(args.processes)
    if not required_processes:
        raise SystemExit("No se encontraron procesos a monitorear (revisá HEARTBEAT_PROCESSES).")

    run_heartbeat(
        required_processes=required_processes,
        interval_hours=max(0.01, args.interval_hours),
        once=args.once,
    )


if __name__ == "__main__":
    main()
