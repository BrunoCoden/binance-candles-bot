# build_dashboard.py
import argparse
import os
import sys
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


try:
    from .config import OUTPUT_PRESETS, resolve_profile
except ImportError:  # ejecución directa
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    if str(CURRENT_DIR.parent) not in sys.path:
        sys.path.append(str(CURRENT_DIR.parent))
    from config import OUTPUT_PRESETS, resolve_profile

DEFAULT_PRICE_PATH = Path(os.getenv("ALERTS_TABLE_CSV_PATH", "alerts_stream.csv"))


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de trades: {path}")
    df = pd.read_csv(path)
    if "EntryTime" in df.columns:
        df["EntryTime"] = pd.to_datetime(df["EntryTime"])
    if "ExitTime" in df.columns:
        df["ExitTime"] = pd.to_datetime(df["ExitTime"])
    return df


def load_price(path: Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    if not path.exists():
        print(f"[DASHBOARD][WARN] Archivo de precios no encontrado: {path}")
        return None
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df.set_index("Timestamp", inplace=True)
    return df


def summarize_trades(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"Total trades": 0}
    wins = (df["Outcome"] == "win").sum()
    losses = (df["Outcome"] == "loss").sum()
    total = len(df)
    pnl_pct_sum = df["PnLPct"].sum() * 100
    pnl_pct_avg = df["PnLPct"].mean() * 100
    winrate = wins / total * 100 if total else 0
    cum = df["PnLPct"].fillna(0).cumsum()
    max_drawdown = cum.min() * 100
    return {
        "Total trades": total,
        "Wins": wins,
        "Losses": losses,
        "Win rate %": f"{winrate:.2f}",
        "Total PnL %": f"{pnl_pct_sum:.2f}",
        "Avg PnL %": f"{pnl_pct_avg:.2f}",
        "Max Drawdown %": f"{max_drawdown:.2f}",
    }


def build_figure(trades: pd.DataFrame, price_df: pd.DataFrame | None):
    if trades.empty:
        raise ValueError("No hay trades para mostrar.")

    rows = 3 if price_df is not None else 2
    specs = [[{"type": "xy"}] for _ in range(rows)]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        specs=specs,
        row_heights=[0.5, 0.3, 0.2] if rows == 3 else [0.6, 0.4],
    )

    row_idx = 1
    if price_df is not None:
        fig.add_trace(
            go.Scatter(
                x=price_df.index,
                y=price_df["Close"],
                name="Close",
                line=dict(color="#222", width=1.2),
                hovertemplate="%{x}<br>Close: %{y:.2f}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )

        long_entries = trades[trades["Direction"] == "long"]
        short_entries = trades[trades["Direction"] == "short"]

        if not long_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_entries["EntryTime"],
                    y=long_entries["EntryPrice"],
                    mode="markers",
                    name="Long Entry",
                    marker=dict(symbol="triangle-up", color="#16a34a", size=9),
                    hovertemplate="Long Entry<br>%{x}<br>%{y:.2f}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=long_entries["ExitTime"],
                    y=long_entries["ExitPrice"],
                    mode="markers",
                    name="Long Exit",
                    marker=dict(symbol="x", color="#16a34a", size=9),
                    hovertemplate="Long Exit<br>%{x}<br>%{y:.2f}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

        if not short_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_entries["EntryTime"],
                    y=short_entries["EntryPrice"],
                    mode="markers",
                    name="Short Entry",
                    marker=dict(symbol="triangle-down", color="#dc2626", size=9),
                    hovertemplate="Short Entry<br>%{x}<br>%{y:.2f}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=short_entries["ExitTime"],
                    y=short_entries["ExitPrice"],
                    mode="markers",
                    name="Short Exit",
                    marker=dict(symbol="x", color="#dc2626", size=9),
                    hovertemplate="Short Exit<br>%{x}<br>%{y:.2f}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

        row_idx += 1

    cum_pct = trades["PnLPct"].fillna(0).cumsum() * 100
    fig.add_trace(
        go.Scatter(
            x=trades["ExitTime"],
            y=cum_pct,
            mode="lines+markers",
            name="PnL acumulado %",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=6),
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
        ),
        row=row_idx,
        col=1,
    )
    fig.update_yaxes(title_text="PnL %", row=row_idx, col=1)
    row_idx += 1

    fig.add_trace(
        go.Histogram(
            x=trades["PnLPct"] * 100,
            nbinsx=20,
            marker=dict(color="#737373"),
            name="Distribución PnL %",
            hovertemplate="%{x:.2f}%<extra></extra>",
        ),
        row=row_idx,
        col=1,
    )
    fig.update_yaxes(title_text="Frecuencia", row=row_idx, col=1)
    fig.update_xaxes(title_text="PnL (%)", row=row_idx, col=1)

    fig.update_layout(
        height=780,
        template="plotly_white",
        title="Dashboard Estrategia Bollinger",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def build_summary_html(summary: dict) -> str:
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary.items())
    return f"""
    <section class="summary">
        <h2>Resumen</h2>
        <table>
            {rows}
        </table>
    </section>
    """


def build_trades_table(trades: pd.DataFrame, limit: int = 50) -> str:
    subset = trades.tail(limit).copy()
    subset["EntryTime"] = subset["EntryTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    subset["ExitTime"] = subset["ExitTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    html_table = subset.to_html(index=False, classes="trades-table")
    return f"""
    <section class="trades">
        <h2>Últimos trades (hasta {limit})</h2>
        {html_table}
    </section>
    """


def render_dashboard(trades_path: Path, price_path: Path | None, html_out: Path, show: bool):
    trades_df = load_trades(trades_path)
    price_df = load_price(price_path)
    summary = summarize_trades(trades_df)

    print("[DASHBOARD] Resumen trades:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    fig = build_figure(trades_df, price_df)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False})

    summary_html = build_summary_html(summary)
    trades_table_html = build_trades_table(trades_df)

    html_out.parent.mkdir(parents=True, exist_ok=True)

    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8" />
    <title>Dashboard Estrategia Bollinger</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #111827;
            color: #f9fafb;
            margin: 0;
            padding: 0 24px 48px;
        }}
        h1 {{
            margin-top: 32px;
            text-align: center;
        }}
        h2 {{
            margin-top: 32px;
            border-left: 4px solid #2563eb;
            padding-left: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            background: #1f2937;
        }}
        th, td {{
            padding: 8px 12px;
            border-bottom: 1px solid #374151;
            text-align: left;
        }}
        th {{
            color: #60a5fa;
            width: 30%;
        }}
        .trades-table th {{
            background: #111827;
        }}
        .trades-table tbody tr:nth-child(even) {{
            background: #111827;
        }}
        .plot-container {{
            margin-top: 32px;
        }}
        a.download {{
            display: inline-block;
            margin-top: 24px;
            padding: 10px 16px;
            background: #2563eb;
            color: white;
            text-decoration: none;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <h1>Dashboard Estrategia Bollinger</h1>
    {summary_html}
    <div class="plot-container">
        {fig_html}
    </div>
    {trades_table_html}
</body>
</html>"""

    html_out.write_text(full_html, encoding="utf-8")
    print(f"[DASHBOARD] HTML generado en {html_out}")

    if show:
        webbrowser.open(html_out.resolve().as_uri())


def main():
    parser = argparse.ArgumentParser(description="Dashboard HTML para trades de la estrategia Bollinger.")
    parser.add_argument("--profile", choices=sorted(OUTPUT_PRESETS.keys()), default=None, help="Preset de salidas (tr o historico).")
    parser.add_argument("--trades", type=str, default=None, help="CSV con trades a visualizar.")
    parser.add_argument("--price", type=str, default=str(DEFAULT_PRICE_PATH), help="CSV con precios (ej. alerts_stream.csv).")
    parser.add_argument("--html", type=str, default=None, help="Archivo HTML de salida.")
    parser.add_argument("--show", action="store_true", help="Abrir el dashboard en el navegador al finalizar.")
    args = parser.parse_args()

    profile = resolve_profile(args.profile)
    preset_paths = OUTPUT_PRESETS[profile]

    trades_path = Path(args.trades) if args.trades else preset_paths["trades"]
    price_path = Path(args.price) if args.price else None
    html_path = Path(args.html) if args.html else preset_paths["dashboard"]

    render_dashboard(trades_path, price_path, html_path, args.show)


if __name__ == "__main__":
    main()
