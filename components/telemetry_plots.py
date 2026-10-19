import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOT_BG = "#0B0D10"
PANEL_BG = "#0E1116"
GRID = "#23262B"
TEXT = "#D7DADE"
MUTED = "#8B9097"
ACCENT = "#FF1801"

_DARK = dict(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT, family="Segoe UI, sans-serif", size=13),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.06, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
)

_ROW_TITLES = ["Speed (km/h)", "Delta (s)", "Throttle (%)", "Brake (%)", "Gear"]


def telemetry_figure(aligned: dict) -> go.Figure:
    colors = aligned.get("colors", {"a": "#1f77b4", "b": "#ff7f0e"})
    names = (aligned["driver_a"], aligned["driver_b"])
    x = aligned["a"]["Distance"].to_numpy()

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=_ROW_TITLES, x_title="Distance (m)",
    )

    traces = [
        (1, "Speed"),
        (3, "Throttle"),
        (4, "Brake"),
        (5, "Gear"),
    ]
    for row, col in traces:
        for key in ("a", "b"):
            fig.add_trace(
                go.Scatter(
                    x=x, y=aligned[key][col],
                    name=names[0 if key == "a" else 1],
                    line=dict(color=colors[key], width=2),
                    legendgroup=key, showlegend=row == 1,
                ),
                row=row, col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=x, y=aligned["delta"]["Delta_s"], name="Delta",
            line=dict(color=ACCENT, width=2),
        ),
        row=2, col=1,
    )

    fig.update_layout(height=760, margin=dict(l=60, r=20, t=40, b=40), **_DARK)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    for ax in fig.select_annotations():
        ax.font.color = MUTED
        ax.font.size = 12
    for i, ax in enumerate(fig.select_yaxes()):
        ax.showgrid = i != 4
        ax.gridcolor = GRID
        ax.zeroline = False
        ax.showline = False
    fig.update_xaxes(showline=False)
    return fig
