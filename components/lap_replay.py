import numpy as np
import plotly.graph_objects as go

from components.telemetry_plots import PANEL_BG, TEXT
from components.track_map import rotate_xy

_OUTLINE_COLOR = "#3A3F46"
_STEP_TARGET = 200  # slider steps; pace scrubbing stays usable above ~200 frames


def _clock_text(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes}:{seconds % 60:04.1f}"


def _hidden_text(name: str) -> str:
    return f"{name} pit / out-lap"


def _frame_caption(frame_index: int, frames: dict, names: list) -> str:
    parts = [f"clock {_clock_text(float(frames['t'][frame_index]))}"]
    for name in names:
        dot = frames["drivers"][name]
        speed = dot["speed"][frame_index]
        if np.isnan(speed):
            parts.append(_hidden_text(name))
        else:
            lap = dot["lap"][frame_index]
            lap_text = f" L{int(lap)}" if lap >= 0 else ""
            parts.append(f"{name}{lap_text} {int(speed)} km/h")
    return "  ·  ".join(parts)


def _frame_hover(frame_index: int, frames: dict, name: str) -> str:
    dot = frames["drivers"][name]
    speed = dot["speed"][frame_index]
    if np.isnan(speed):
        return "pit / out-lap"
    lap = dot["lap"][frame_index]
    return f"Lap {int(lap)} · {_clock_text(float(frames['t'][frame_index]))} · {int(speed)} km/h"


def _marker_style(color: str) -> dict:
    return dict(size=10, color=color, symbol="circle", line=dict(width=0))


def replay_figure(
    frames: dict,
    curves: dict,
    colors: dict,
    corners=None,
    rotation_deg: float = 0.0,
    pace: float = 1.0,
) -> go.Figure:
    names = list(frames["drivers"])
    first = curves.get(names[0])
    if first is None:
        raise ValueError("No curve available for the track outline")
    x, y = rotate_xy(first["x"], first["y"], rotation_deg)
    n = frames["n_frames"]
    span_s = frames["span"][1] - frames["span"][0]
    frame_ms = max(10, int(span_s * 1000 / n / pace))

    fig = go.Figure()
    # Moving traces first; frame data updates by position for these indexes.
    for name in names:
        fig.add_trace(
            go.Scatter(
                x=[], y=[], mode="markers", name=name,
                marker=_marker_style(colors.get(name, TEXT)),
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            )
        )
    caption_x = float(x.min())
    caption_y = float(y.max())
    span_y = float(y.max() - y.min())
    fig.add_trace(
        go.Scatter(
            x=[caption_x], y=[caption_y + 0.06 * span_y], mode="text", name="clock",
            text=[""], textfont=dict(size=13, color=TEXT),
            textposition="top left", hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=_OUTLINE_COLOR, width=10),
            hoverinfo="skip", showlegend=False,
        )
    )
    if corners is not None and len(corners):
        cx, cy = rotate_xy(corners["X"].to_numpy(), corners["Y"].to_numpy(), rotation_deg)
        labels = [
            f"T{int(number)}{letter}".strip()
            for number, letter in zip(corners["Number"], corners["Letter"])
        ]
        fig.add_trace(
            go.Scatter(
                x=cx, y=cy, mode="markers+text", text=labels,
                textposition="top center",
                marker=dict(size=7, color=TEXT, symbol="circle-open"),
                textfont=dict(size=11, color=TEXT),
                name="Corners", hovertemplate="%{text}<extra></extra>", showlegend=False,
            )
        )

    animated = []
    for i in range(n):
        data = []
        for name in names:
            dot = frames["drivers"][name]
            data.append(
                go.Scatter(
                    x=[float(dot["x"][i])], y=[float(dot["y"][i])], mode="markers", name=name,
                    marker=_marker_style(colors.get(name, TEXT)),
                    hovertemplate="%{customdata}<extra></extra>",
                    showlegend=False,
                    customdata=[_frame_hover(i, frames, name)],
                )
            )
        data.append(
            go.Scatter(
                x=[caption_x], y=[caption_y + 0.06 * span_y], mode="text", name="clock",
                text=[_frame_caption(i, frames, names)],
                textfont=dict(size=13, color=TEXT),
                textposition="top left", hoverinfo="skip", showlegend=False,
            )
        )
        animated.append(go.Frame(data=data, name=str(i)))
    fig.frames = animated

    step_stride = max(1, n // _STEP_TARGET)
    steps = [
        dict(
            method="animate",
            label="",
            args=[[str(i)], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
        )
        for i in range(0, n, step_stride)
    ]
    buttons = [
        dict(
            label="Play", method="animate",
            args=[
                None,
                dict(
                    frame=dict(duration=frame_ms, redraw=True),
                    transition=dict(duration=0),
                    mode="immediate",
                    fromcurrent=True,
                ),
            ],
        ),
        dict(
            label="Pause", method="animate",
            args=[[None], dict(mode="immediate", frame=None, transition=None)],
        ),
        dict(
            label="Restart", method="animate",
            args=[[str(0)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
        ),
    ]

    fig.update_layout(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG, font=dict(color=TEXT),
        showlegend=False, margin=dict(l=10, r=10, t=16, b=10),
        updatemenus=[
            dict(
                type="buttons", direction="right", x=0.0, y=1.0,
                buttons=buttons, bgcolor=PANEL_BG,
                font=dict(color=TEXT), showactive=False,
            )
        ],
        sliders=[dict(active=0, currentvalue=dict(visible=False), pad=dict(t=10), steps=steps)],
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig
