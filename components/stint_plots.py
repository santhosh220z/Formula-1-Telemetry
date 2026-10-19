import numpy as np
import plotly.graph_objects as go

from components.telemetry_plots import GRID, MUTED, PANEL_BG, TEXT


def stint_evolution_figure(laps, stats, colors) -> go.Figure:
    fig = go.Figure()
    grouped = laps.groupby(["Driver", "Stint"], observed=True)
    for (driver, stint), group in grouped:
        color = colors.get(driver, MUTED)
        compound = str(group["Compound"].iloc[0])
        label = f"{driver} · {compound} S{int(stint)}"
        fig.add_trace(
            go.Scatter(
                x=group["TyreLife"], y=group["LapTime_s"],
                mode="lines+markers", name=label, legendgroup=label,
                line=dict(color=color, width=2), marker=dict(size=5),
            )
        )
        fit = stats[(stats["driver"] == driver) & (stats["stint"] == int(stint))]
        if len(fit):
            row = fit.iloc[0]
            xs = np.linspace(group["TyreLife"].min(), group["TyreLife"].max(), 20)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=row["base_s"] + row["deg_slope_s_per_lap"] * xs,
                    mode="lines", showlegend=False, legendgroup=label,
                    line=dict(color=color, width=1.5, dash="dot"),
                )
            )

    fig.update_layout(
        height=430, margin=dict(l=55, r=18, t=16, b=40),
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT, family="Segoe UI, sans-serif"),
        legend=dict(orientation="h", y=1.08, yanchor="bottom", bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(title="Tyre life (laps)", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title="Lap time (s)", gridcolor=GRID, zeroline=False)
    return fig
