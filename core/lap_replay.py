import numpy as np
import pandas as pd

_FRAME_SECONDS = 0.6
_MIN_FRAMES = 90
_MAX_FRAMES = 600


def _window(curve: dict, index: int) -> tuple[float, float]:
    return curve["lap_starts"][index], curve["lap_ends"][index]


def _aligned(arrays: list, owners: list, order: np.ndarray, uniq_mask: np.ndarray):
    stack = np.concatenate(arrays)[order]
    owner = np.concatenate(owners)[order]
    return stack[uniq_mask], owner[uniq_mask]


def session_curves(laps: list[tuple[int, pd.DataFrame]]) -> dict:
    """Blend per-lap telemetry profiles into one session-relative curve.

    Each profile needs Time (session-relative seconds), X, Y, Speed columns.
    Returns t/x/y/speed arrays plus per-lap visibility windows.
    """
    times, xs, ys, speeds, owners = [], [], [], [], []
    windows = []
    for lap_no, profile in laps:
        t = profile["Time"].to_numpy(dtype=float)
        x = profile["X"].to_numpy(dtype=float)
        y = profile["Y"].to_numpy(dtype=float)
        speed = profile["Speed"].to_numpy(dtype=float)
        keep = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(speed)
        if not keep.any():
            continue
        t, x, y, speed = t[keep], x[keep], y[keep], speed[keep]
        times.append(t)
        xs.append(x)
        ys.append(y)
        speeds.append(speed)
        owners.append(np.full(t.size, lap_no, dtype=int))
        windows.append((float(t.min()), float(t.max()), lap_no))
    if not times:
        raise ValueError("No timed laps with position telemetry")
    t_all = np.concatenate(times)
    order = np.argsort(t_all, kind="stable")
    t = np.sort(t_all)
    uniq_mask = np.concatenate([[True], np.diff(t) > 0])

    x, lap_of_sample = _aligned(xs, owners, order, uniq_mask)
    y, _ = _aligned(ys, owners, order, uniq_mask)
    speed, _ = _aligned(speeds, owners, order, uniq_mask)
    t = t[uniq_mask]
    windows.sort()
    return {
        "t": t,
        "x": x,
        "y": y,
        "speed": speed,
        "lap_of_sample": lap_of_sample,
        "lap_starts": [w[0] for w in windows],
        "lap_ends": [w[1] for w in windows],
        "lap_numbers": [w[2] for w in windows],
    }


def crop_curve(curve: dict, t0: float, t1: float) -> dict:
    mask = (curve["t"] >= t0) & (curve["t"] <= t1)
    cropped = {key: value[mask] for key, value in curve.items() if isinstance(value, np.ndarray)}
    starts, ends, numbers = [], [], []
    for start, end, number in zip(
        curve["lap_starts"], curve["lap_ends"], curve["lap_numbers"]
    ):
        span_start, span_end = max(start, t0), min(end, t1)
        if span_end < span_start:
            continue
        starts.append(span_start)
        ends.append(span_end)
        numbers.append(number)
    cropped["lap_starts"] = starts
    cropped["lap_ends"] = ends
    cropped["lap_numbers"] = numbers
    return cropped


def span_of(curves: dict[str, dict]) -> tuple[float, float]:
    t0 = min(curve["t"].min() for curve in curves.values())
    t1 = max(curve["t"].max() for curve in curves.values())
    return float(t0), float(t1)


def _auto_frames(span: float) -> int:
    return int(np.clip(np.ceil(span / _FRAME_SECONDS), _MIN_FRAMES, _MAX_FRAMES))


def replay_frames(curves: dict[str, dict], n_frames: int | None = None) -> dict:
    t0, t1 = span_of(curves)
    n = n_frames if n_frames is not None else _auto_frames(t1 - t0)
    clock = np.linspace(t0, t1, n)
    drivers = {}
    for name, curve in curves.items():
        visible = np.zeros(n, dtype=bool)
        lap_index = np.full(n, -1, dtype=int)
        for index, (start, end) in enumerate(zip(curve["lap_starts"], curve["lap_ends"])):
            inside = (clock >= start) & (clock <= end)
            visible |= inside
            lap_index[inside] = index
        x = np.interp(clock, curve["t"], curve["x"])
        y = np.interp(clock, curve["t"], curve["y"])
        speed = np.interp(clock, curve["t"], curve["speed"])
        hidden = ~visible
        x[hidden] = np.nan
        y[hidden] = np.nan
        speed[hidden] = np.nan
        numbers = curve["lap_numbers"]
        drivers[name] = {
            "x": x,
            "y": y,
            "speed": speed,
            "lap": np.array([numbers[i] if i >= 0 else -1 for i in lap_index], dtype=int),
        }
    return {"t": clock, "span": (t0, t1), "n_frames": n, "drivers": drivers}
