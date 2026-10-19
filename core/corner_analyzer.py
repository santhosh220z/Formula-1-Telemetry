import numpy as np
import pandas as pd

WINDOW = (-100.0, 100.0)
BRAKE_GRAD_PER_M = -0.05
THROTTLE_PICKUP_PCT = 40.0
SUSTAINED_POINTS = 5
TIE_TOLERANCE_S = 0.02


def locate_corners(aligned: dict, corners: pd.DataFrame) -> pd.DataFrame:
    """Project circuit-info corner markers (X/Y) onto the telemetry path -> lap distance."""
    frame = aligned["a"]
    px = frame["X"].to_numpy(dtype=float)
    py = frame["Y"].to_numpy(dtype=float)
    pdist = frame["Distance"].to_numpy(dtype=float)
    rows = []
    for _, marker in corners.iterrows():
        d2 = (px - float(marker["X"])) ** 2 + (py - float(marker["Y"])) ** 2
        i = int(np.argmin(d2))
        rows.append(
            {
                "Number": int(marker["Number"]),
                "Letter": str(marker.get("Letter", "") or ""),
                "X": float(marker["X"]),
                "Y": float(marker["Y"]),
                "Distance": float(pdist[i]),
            }
        )
    return pd.DataFrame(rows)


def _first_window(values: np.ndarray, condition: np.ndarray) -> float:
    """Distance of the first point starting SUSTAINED_POINTS consecutive True values."""
    kernel = np.ones(SUSTAINED_POINTS, dtype=int)
    runs = np.convolve(condition.astype(int), kernel, mode="valid") == SUSTAINED_POINTS
    idx = np.argmax(runs) if runs.any() else -1
    return idx if idx >= 0 else np.nan


def _driver_window_metrics(frame: pd.DataFrame, start: float, end: float) -> dict:
    distance = frame["Distance"].to_numpy(dtype=float)
    speed = frame["Speed_ms"].to_numpy(dtype=float)
    in_window = (distance >= start) & (distance <= end)
    win_idx = np.flatnonzero(in_window)
    i_min = int(win_idx[int(np.argmin(speed[win_idx]))])
    grad = np.gradient(speed, distance)
    smoothed = np.convolve(grad, np.full(SUSTAINED_POINTS, 1 / SUSTAINED_POINTS), mode="valid")
    pre_min = smoothed[: max(1, i_min - 3)]
    brake_runs = pre_min < BRAKE_GRAD_PER_M
    brake_idx = np.argmax(brake_runs) if brake_runs.any() else -1
    throttle_runs = np.convolve(
        frame["Throttle"].to_numpy(dtype=float)[i_min:] > THROTTLE_PICKUP_PCT,
        np.ones(SUSTAINED_POINTS, dtype=float), mode="valid",
    )
    pickup_idx = int(np.argmax(throttle_runs == SUSTAINED_POINTS)) + i_min if (throttle_runs == SUSTAINED_POINTS).any() else -1
    return {
        "v_min_ms": float(speed[i_min]),
        "v_min_dist": float(distance[i_min]),
        "brake_onset_dist": float(distance[brake_idx]) if brake_idx >= 0 else np.nan,
        "throttle_pickup_dist": float(distance[pickup_idx]) if pickup_idx >= 0 else np.nan,
    }


def corner_window_metrics(aligned: dict, corner_distance: float, window=WINDOW) -> dict:
    grid = aligned["grid"]
    start = max(grid[0], corner_distance + window[0])
    end = min(grid[-1], corner_distance + window[1])
    if end <= start:
        raise ValueError("Corner window does not overlap the telemetry grid")

    metrics = {}
    for key in ("a", "b"):
        metrics[key] = _driver_window_metrics(aligned[key], start, end)

    delta = aligned["delta"]
    dvals = delta["Distance"].to_numpy(dtype=float)
    i_start = int(np.searchsorted(dvals, start))
    i_end = max(i_start + 1, int(np.searchsorted(dvals, end, side="right")) - 1)
    dvals_arr = delta["Delta_s"].to_numpy(dtype=float)
    corner_delta = float(dvals_arr[i_end] - dvals_arr[i_start])
    faster = np.nan
    if corner_delta > TIE_TOLERANCE_S:
        faster = aligned["driver_a"]
    elif corner_delta < -TIE_TOLERANCE_S:
        faster = aligned["driver_b"]

    return {**{f"{key}_{k}": v for key in ("a", "b") for k, v in metrics[key].items()},
            "corner_delta_s": corner_delta, "faster": faster}


def corner_breakdown(aligned: dict, corners: pd.DataFrame, window=WINDOW) -> pd.DataFrame:
    located = locate_corners(aligned, corners)
    rows = []
    for _, corner in located.iterrows():
        m = corner_window_metrics(aligned, corner["Distance"], window)
        rows.append(
            {
                "corner": int(corner["Number"]),
                "letter": corner["Letter"],
                "corner_distance": float(corner["Distance"]),
                **m,
            }
        )
    return pd.DataFrame(rows)


def microsector_advantage(aligned: dict, n: int = 10) -> pd.DataFrame:
    grid = aligned["grid"]
    edges = np.linspace(grid[0], grid[-1], n + 1)
    ta = np.interp(edges, aligned["a"]["Distance"], aligned["a"]["Time"])
    tb = np.interp(edges, aligned["b"]["Distance"], aligned["b"]["Time"])
    adv = (tb[1:] - tb[:-1]) - (ta[1:] - ta[:-1])
    labels = []
    for a in adv:
        if a > TIE_TOLERANCE_S:
            labels.append(aligned["driver_a"])
        elif a < -TIE_TOLERANCE_S:
            labels.append(aligned["driver_b"])
        else:
            labels.append(np.nan)
    return pd.DataFrame(
        {
            "seg": np.arange(1, n + 1),
            "start": edges[:-1],
            "end": edges[1:],
            "adv_s": adv,
            "faster": labels,
        }
    )
