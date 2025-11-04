# Backtests de la estrategia Bollinger

Este directorio reúne las utilidades necesarias para ejecutar backtests de la estrategia de Bandas de Bollinger contra datos de Binance USD‑M Futures. Los flujos cubren dos perfiles de salida:

- **TR**: backtest corto para trading intradía (carpeta `backtestTR`).
- **Histórico**: corridas extensas para análisis de largo plazo (carpeta `backtestHistorico`).

Ambos perfiles comparten el mismo motor (`run_backtest.py`) y las rutas/alineaciones de archivos se controlan desde `config.py` y variables de entorno.

## Requisitos previos

- Python 3.11+ recomendado (el proyecto usa pandas, numpy, plotly, etc.).
- Dependencias instaladas: `pip install -r requirements.txt`.
- Variables de entorno cargadas (por ejemplo `source .venv/bin/activate && set -a && source .env && set +a`) para que `velas.py`, `trade_logger.py` y los backtests reciban el símbolo, intervalos, credenciales y rutas de salida.
- Acceso HTTPS al endpoint de Binance (`BINANCE_UM_BASE_URL`, por defecto `https://fapi.binance.com`).

## Archivos clave

- `run_backtest.py`: motor que descarga velas, genera señales Bollinger, simula entradas/salidas y guarda resultados.
- `build_dashboard.py`: genera un dashboard HTML con métricas, gráfico y detalle de operaciones.
- `config.py`: define perfiles `tr` y `historico`, junto con las rutas de CSV/PNG/HTML (pueden sobrescribirse vía variables como `STRAT_BACKTEST_TRADES_PATH`).
- `backtestTR/` y `backtestHistorico/`: carpetas destino para cada perfil (CSV de trades, gráfico y dashboard).

## Ejecutar el backtest

1. **Activar entorno** (si aplica):
   ```bash
   source .venv/bin/activate
   set -a && source .env && set +a  # opcional pero recomendado
   ```

2. **Lanzar el backtest**:

   - Perfil TR (últimas semanas, salidas en `backtest/backtestTR/`):
     ```bash
     python backtest/run_backtest.py --profile tr --weeks 2
     ```
     Ajustá `--weeks` según la ventana que quieras analizar (si lo omitís, usa `BACKTEST_STREAM_BARS`).

   - Perfil Histórico (meses de datos, salidas en `backtest/backtestHistorico/`):
     ```bash
     python backtest/run_backtest.py --profile historico --months 6
     ```
     También podés fijar fechas exactas:
     ```bash
     python backtest/run_backtest.py --profile historico --start 2024-01-01T00:00:00Z --end 2024-06-30T23:59:59Z
     ```

   Durante la ejecución se imprime la comisión estimada en Binance, el rango temporal efectivo y un resumen con métricas (trades totales, win rate, PnL, drawdown, fees).

### Parámetros útiles

`run_backtest.py` acepta varias banderas para ajustar la corrida:

- `--stream-bars`: cantidad base de velas a descargar (default `BACKTEST_STREAM_BARS`).
- `--profile {tr,historico}`: selecciona el preset de rutas definido en `config.py`.
- `--trades-out / --plot-out`: rutas de salida personalizadas para CSV/PNG.
- `--weeks` o `--months`: rango relativo hacia atrás (usar solo uno).
- `--start / --end`: fechas ISO8601 (UTC) para rango absoluto.
- `--show`: abre la figura de Matplotlib al terminar si `matplotlib` está disponible.

> Nota: el motor usa las mismas Bandas de Bollinger configuradas para el watcher (`BB_LENGTH`, `BB_MULT`, `BB_DIRECTION`, `STREAM_INTERVAL`, etc.), por lo que cualquier cambio en `.env` impactará tanto las señales en vivo como el backtest.

## Visualización y dashboards

Una vez generado el CSV de trades podés construir el dashboard interactivo:

```bash
python backtest/build_dashboard.py --profile tr --price alerts_stream.csv
```

- `--profile` funciona igual que en el backtest; si no lo indicás usa el valor por defecto (`BACKTEST_PROFILE` o `tr`).
- `--trades` permite elegir un CSV alternativo (por ejemplo una corrida histórica guardada en otro directorio).
- `--price` es opcional, pero al pasar el CSV de precios (`alerts_stream.csv` u otro con columnas `Timestamp` y `Close`) se superpone la curva de precios con las entradas/salidas.
- `--html` define el destino del dashboard (default según preset).
- `--show` abre automáticamente el HTML en el navegador.

Los dashboards incluyen resumen estadístico, PnL acumulado, histograma de rendimiento y tablas con los últimos trades/operaciones. El archivo resultante se guarda en `backtest/backtestTR/dashboard.html` o `backtest/backtestHistorico/dashboard.html` según el perfil elegido.

### Listener para minuto exacto de fills

Si querés capturar el minuto exacto en que se ejecuta una orden pendiente, corré el listener dedicado (opera sobre el mismo `realtime_state.json` que usa el backtest en vivo):

```bash
python backtest/order_fill_listener.py --profile tr
```

- Monitorea las órdenes con `status=pending` y consulta velas de 1 minuto para detectar el primer cruce del precio objetivo.
- Actualiza el estado a `open` con el timestamp de esa vela (UTC) y mantiene la misma lógica de SL/TP definida por la estrategia.
- Parámetros opcionales:
  - `--poll-seconds` (default 15) ajusta la frecuencia de consulta.
  - `--tolerance` permite sumar una tolerancia absoluta al match del precio.
  - `--lookback-minutes` define la ventana de búsqueda al reconstruir la vela que ejecutó la orden.

Mantenelo corriendo junto al watcher de señales si necesitás una simulación intradía con precisión de minuto.

## Consejos y buenas prácticas

- Confirmá que `alerts_stream.csv` esté poblado si querés overlay de precios en el dashboard; el watcher `watcher_alertas.py` lo genera automáticamente.
- Para corridas históricas largas, aumentar `PAGINATE_PAGE_LIMIT` y `PAGE_SLEEP_SEC` puede acelerar las descargas sin exceder límites de Binance.
- Si necesitás replicar los resultados en otro equipo, copiá el `.env` (sin credenciales sensibles) y las carpetas `backtestTR/` / `backtestHistorico/`.
- El script maneja comisiones usando el `takerCommissionRate` que expone Binance; si falla la consulta, aplica el fallback 0.0005. Podés forzar una tarifa fija exportando `STRAT_FEE_RATE` antes de ejecutar el backtest.

Con estos pasos deberías poder generar y analizar tanto corridas recientes (TR) como estudios históricos completos de la estrategia.
