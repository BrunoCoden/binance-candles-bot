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
    from velas import SYMBOL_DISPLAY, STREAM_INTERVAL
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    if str(PARENT_DIR) not in sys.path:
        sys.path.append(str(PARENT_DIR))
    from velas import SYMBOL_DISPLAY, STREAM_INTERVAL

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

LOGO_SVG = """
<svg width=\"72\" height=\"72\" viewBox=\"0 0 120 120\" xmlns=\"http://www.w3.org/2000/svg\">
  <rect x=\"0\" y=\"0\" width=\"120\" height=\"120\" rx=\"18\" fill=\"#111827\" stroke=\"#2563eb\" stroke-width=\"6\"/>
  <path d=\"M15 60 C30 40, 55 20, 80 45 S115 100, 105 105\" stroke=\"#22d3ee\" stroke-width=\"6\" fill=\"none\"/>
  <path d=\"M15 80 C40 65, 65 50, 90 70\" stroke=\"#a855f7\" stroke-width=\"6\" fill=\"none\" opacity=\"0.8\"/>
  <circle cx=\"78\" cy=\"46\" r=\"8\" fill=\"#facc15\" stroke=\"#facc15\"/>
</svg>
"""


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
    total_fees = df.get("Fees", pd.Series(dtype=float)).sum()
    return {
        "Total trades": total,
        "Wins": wins,
        "Losses": losses,
        "Win rate %": f"{winrate:.2f}",
        "Total PnL %": f"{pnl_pct_sum:.2f}",
        "Avg PnL %": f"{pnl_pct_avg:.2f}",
        "Max Drawdown %": f"{max_drawdown:.2f}",
        "Total Fees": f"{total_fees:.2f}",
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


def build_operations_table(trades: pd.DataFrame, limit: int = 15) -> str:
    columns = [
        ("EntryTime", "Entrada"),
        ("Direction", "Dirección"),
        ("EntryPrice", "Precio Entrada"),
        ("ExitTime", "Salida"),
        ("ExitPrice", "Precio Salida"),
        ("Outcome", "Resultado"),
        ("PnLAbs", "PnL"),
        ("PnLPct", "PnL %"),
        ("Fees", "Fees"),
    ]
    subset = trades.tail(limit).copy()
    subset["EntryTimeFmt"] = subset["EntryTime"].dt.strftime("%d-%m %H:%M")
    subset["ExitTimeFmt"] = subset["ExitTime"].dt.strftime("%d-%m %H:%M")
    rows_html = []
    for _, row in subset.iterrows():
        pnl_pct = row.get("PnLPct", 0) * 100 if pd.notna(row.get("PnLPct")) else 0
        cells = [
            f"<td>{row['EntryTimeFmt']}</td>",
            f"<td class='dir {row['Direction']}'>{row['Direction'].upper()}</td>",
            f"<td>{row['EntryPrice']:.2f}</td>",
            f"<td>{row['ExitTimeFmt']}</td>",
            f"<td>{row['ExitPrice']:.2f}</td>",
            f"<td class='result {row['Outcome']}'>{row['Outcome'].upper()}</td>",
            f"<td>{row['PnLAbs']:.2f}</td>",
            f"<td>{pnl_pct:.2f}%</td>",
            f"<td>{row.get('Fees', 0.0):.2f}</td>",
        ]
        rows_html.append("".join(cells))

    header_cells = "".join(f"<th>{label}</th>" for _, label in columns)
    body_rows = "".join(f"<tr>{row}</tr>" for row in rows_html)

    return f"""
    <section class="ops">
        <h2>Detalle Operativo Reciente</h2>
        <table class="ops-table">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{body_rows}</tbody>
        </table>
    </section>
    """


def render_dashboard(trades_path: Path, price_path: Path | None, html_out: Path, show: bool, profile: str):
    trades_df = load_trades(trades_path)
    price_df = load_price(price_path)
    summary = summarize_trades(trades_df)

    print("[DASHBOARD] Resumen trades:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    fig = build_figure(trades_df, price_df)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False})

    summary_html = build_summary_html(summary)
    ops_table_html = build_operations_table(trades_df)
    trades_table_html = build_trades_table(trades_df)

    html_out.parent.mkdir(parents=True, exist_ok=True)

    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8" />
    <title>Dashboard Estrategia Bollinger</title>
    <style>
        body {{
            font-family: 'Inter', Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 32px 24px 56px;
        }}
        .hero {{
            display: flex;
            align-items: center;
            gap: 24px;
            margin-bottom: 24px;
        }}
        .hero .logo {{
            flex-shrink: 0;
        }}
        .hero h1 {{
            margin: 0;
            font-size: 1.9rem;
        }}
        .hero p {{
            margin: 6px 0 0;
            color: #94a3b8;
        }}
        h2 {{
            margin-top: 32px;
            border-left: 4px solid #2563eb;
            padding-left: 12px;
            font-size: 1.3rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            background: #1e293b;
            border-radius: 10px;
            overflow: hidden;
        }}
        th, td {{
            padding: 10px 14px;
            border-bottom: 1px solid #334155;
            text-align: left;
            font-size: 0.95rem;
        }}
        th {{
            color: #60a5fa;
            background: rgba(37, 99, 235, 0.12);
        }}
        .trades-table tbody tr:nth-child(even),
        .ops-table tbody tr:nth-child(even) {{
            background: rgba(15, 23, 42, 0.6);
        }}
        .plot-container {{
            margin-top: 32px;
        }}
        .dir.long {{
            color: #22c55e;
            font-weight: 600;
        }}
        .dir.short {{
            color: #f87171;
            font-weight: 600;
        }}
        .result.win {{
            color: #4ade80;
            font-weight: 600;
        }}
        .result.loss {{
            color: #f87171;
            font-weight: 600;
        }}
        .result.flat {{
            color: #fbbf24;
            font-weight: 600;
        }}
        @media (max-width: 768px) {{
            .hero {{
                flex-direction: column;
                align-items: flex-start;
            }}
            .hero .logo {{
                margin-bottom: 8px;
            }}
            th, td {{
                font-size: 0.85rem;
            }}
        }}
    </style>
</head>
<body>
    <section class="hero">
        <div class="logo">{LOGO_SVG}</div>
        <div>
            <h1>Dashboard Estrategia Bollinger</h1>
            <p>{SYMBOL_DISPLAY} · Intervalo {STREAM_INTERVAL} · Perfil {profile.upper()}</p>
        </div>
    </section>
    {summary_html}
    <div class="plot-container">
        {fig_html}
    </div>
    {ops_table_html}
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

    render_dashboard(trades_path, price_path, html_path, args.show, profile)


if __name__ == "__main__":
    main()
