import numpy as np
import plotly.graph_objects as go

from components.telemetry_plots import PANEL_BG, TEXT

_TIE_COLOR = "#5A6067"


def _rotate(x, y, degrees):
    angle = np.deg2rad(degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def _is_nan(value) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return value is None


def _seg_color(faster, aligned: dict, colors: dict) -> str:
    if _is_nan(faster):
        return _TIE_COLOR
    return colors["a"] if faster == aligned["driver_a"] else colors["b"]


def track_map_figure(aligned: dict, corners=None, rotation_deg: float = 0.0, microsectors=None) -> go.Figure:
    colors = aligned.get("colors", {"a": "#1f77b4", "b": "#ff7f0e"})
    frame = aligned["a"]
    x, y = _rotate(frame["X"].to_numpy(), frame["Y"].to_numpy(), rotation_deg)
    dist = frame["Distance"].to_numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color="#3A3F46", width=10),
            hoverinfo="skip", showlegend=False,
        )
    )

    if microsectors is not None and len(microsectors):
        for _, seg in microsectors.iterrows():
            mask = (dist >= seg["start"]) & (dist < seg["end"])
            if not mask.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=x[mask], y=y[mask], mode="lines",
                    line=dict(color=_seg_color(seg["faster"], aligned, colors), width=5),
                    customdata=np.full(int(mask.sum()), seg["adv_s"]),
                    hovertemplate="Mini-sector <%{customdata:.2f}s vs slower driver><extra></extra>",
                    showlegend=False,
                )
            )

    if corners is not None and len(corners):
        cx, cy = _rotate(corners["X"].to_numpy(), corners["Y"].to_numpy(), rotation_deg)
        letters = corners["Letter"].fillna("").astype(str)
        labels = [
            f"T{int(n)}{letter}".strip()
            for n, letter in zip(corners["Number"], letters)
        ]
        fig.add_trace(
            go.Scatter(
                x=cx, y=cy, mode="markers+text", text=labels,
                textposition="top center",
                marker=dict(size=7, color=TEXT, symbol="circle-open"),
                textfont=dict(size=11, color=TEXT),
                name="Corners",
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT), showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig
