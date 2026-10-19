import plotly.graph_objects as go

from components.telemetry_plots import ACCENT, MUTED, PANEL_BG, TEXT


def _table(header, columns, column_font_colors, height, name="tbl"):
    fig = go.Figure(
        go.Table(
            header=dict(
                values=header,
                fill_color="#171A20", font=dict(color=MUTED, size=12),
                align="center", height=34,
            ),
            cells=dict(
                values=columns,
                fill_color=PANEL_BG, font=dict(color=column_font_colors, size=13),
                align="center", height=30,
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor=PANEL_BG,
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
    )
    return fig


def corner_table(breakdown) -> go.Figure:
    b = breakdown
    labels = (b["corner"].astype("str") + b["letter"].fillna("")).tolist()
    n_rows = max(len(b), 1)
    winner_colors = [ACCENT] + [TEXT] * (len(b.columns) - 3) + [TEXT]
    header = ["Corner", "Δ (s)", "Winner", "V_min A (km/h)", "V_min B (km/h)",
              "Brake A (m)", "Brake B (m)", "Throttle A (m)", "Throttle B (m)"]
    columns = [
        labels,
        b["corner_delta_s"].round(2),
        b["faster"],
        (b["a_v_min_ms"] * 3.6).round(0),
        (b["b_v_min_ms"] * 3.6).round(0),
        b["a_brake_onset_dist"].round(0),
        b["b_brake_onset_dist"].round(0),
        b["a_throttle_pickup_dist"].round(0),
        b["b_throttle_pickup_dist"].round(0),
    ]
    # column colors: winner column highlighted, delta column muted
    colors = [TEXT, MUTED, ACCENT] + [TEXT] * 6
    return _table(header, columns, colors, min(700, 90 + 30 * n_rows))


def stint_table(stats) -> go.Figure:
    s = stats
    header = ["Driver", "Stint", "Compound", "Laps", "Base (s)", "Deg (s/lap)", "Deg total (s)"]
    columns = [
        s["driver"].tolist(),
        s["stint"].astype(str).tolist(),
        s["compound"].astype(str).tolist(),
        s["laps"].astype(str).tolist(),
        s["base_s"].round(2).tolist(),
        s["deg_slope_s_per_lap"].round(3).tolist(),
        s["deg_total_s"].round(2).tolist(),
    ]
    colors = [TEXT] * len(header)
    return _table(header, columns, colors, min(700, 90 + 30 * max(len(s), 1)))
