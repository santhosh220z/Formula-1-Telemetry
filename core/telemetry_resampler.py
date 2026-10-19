import numpy as np
import pandas as pd

CHANNELS = ["Speed", "Speed_ms", "Throttle", "Brake", "Gear", "X", "Y"]

_MAX_GRID_POINTS = 20000


def distance_from_speed(time: np.ndarray, speed_ms: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral of speed: distance at each time sample.

    Monotonic by construction because speeds are non-negative.
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(speed_ms, dtype=float)
    steps = 0.5 * (v[:-1] + v[1:]) * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(steps)])


def common_grid(dist_a: np.ndarray, dist_b: np.ndarray, step: float = 1.0) -> np.ndarray:
    start = max(float(dist_a[0]), float(dist_b[0]))
    end = min(float(dist_a[-1]), float(dist_b[-1]))
    if end <= start:
        raise ValueError("Drivers' laps do not overlap in distance")
    n = int(np.ceil((end - start) / step)) + 1
    n = min(n, _MAX_GRID_POINTS)
    return np.linspace(start, end, n)


def resample_on_distance(profile: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    out = {"Distance": grid}
    distance = profile["Distance"].to_numpy(dtype=float)
    for col in profile.columns:
        if col == "Distance":
            continue
        values = profile[col].to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(distance)
        out[col] = np.interp(grid, distance[mask], values[mask])
    if "Gear" in out:
        out["Gear"] = np.round(out["Gear"])
    return pd.DataFrame(out)


def time_at_distance(grid: np.ndarray, dist: np.ndarray, time: np.ndarray) -> np.ndarray:
    return np.interp(grid, dist, time)


def delta_time(
    grid: np.ndarray,
    dist_a: np.ndarray,
    time_a: np.ndarray,
    dist_b: np.ndarray,
    time_b: np.ndarray,
) -> np.ndarray:
    """Positive result means driver b is slower at that distance."""
    return time_at_distance(grid, dist_b, time_b) - time_at_distance(grid, dist_a, time_a)


def align_profiles(
    profile_a: pd.DataFrame,
    profile_b: pd.DataFrame,
    driver_a: str = "A",
    driver_b: str = "B",
) -> dict:
    grid = common_grid(
        profile_a["Distance"].to_numpy(dtype=float),
        profile_b["Distance"].to_numpy(dtype=float),
    )
    a = resample_on_distance(profile_a, grid)
    b = resample_on_distance(profile_b, grid)
    delta = b["Time"].to_numpy(dtype=float) - a["Time"].to_numpy(dtype=float)
    return {
        "a": a,
        "b": b,
        "delta": pd.DataFrame({"Distance": grid, "Delta_s": delta}),
        "grid": grid,
        "driver_a": driver_a,
        "driver_b": driver_b,
    }


def extract_lap_profile(lap) -> pd.DataFrame:
    tel = lap.get_telemetry()
    time_s = tel["Time"].dt.total_seconds().to_numpy(dtype=float)
    speed_kmh = tel["Speed"].to_numpy(dtype=float)
    speed_ms = speed_kmh / 3.6
    return pd.DataFrame(
        {
            "Time": time_s,
            "Distance": distance_from_speed(time_s, speed_ms),
            "Speed": speed_kmh,
            "Speed_ms": speed_ms,
            "Throttle": tel["Throttle"].to_numpy(dtype=float),
            "Brake": tel["Brake"].to_numpy(dtype=float),
            "Gear": tel["nGear"].to_numpy(dtype=float),
            "X": tel["X"].to_numpy(dtype=float),
            "Y": tel["Y"].to_numpy(dtype=float),
        }
    )


def build_aligned_telemetry(session, driver_a: str, driver_b: str, lap_a=None, lap_b=None) -> dict:
    from core.session_manager import best_lap

    lap_a = lap_a if lap_a is not None else best_lap(session, driver_a)
    lap_b = lap_b if lap_b is not None else best_lap(session, driver_b)
    return align_profiles(
        extract_lap_profile(lap_a), extract_lap_profile(lap_b), driver_a=driver_a, driver_b=driver_b
    )
