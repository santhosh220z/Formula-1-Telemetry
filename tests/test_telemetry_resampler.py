import numpy as np
import pandas as pd


def const_profile(v: float, duration: float, dt: float = 0.1) -> pd.DataFrame:
    t = np.arange(0, duration, dt)
    n = len(t)
    return pd.DataFrame(
        {
            "Time": t,
            "Distance": np.concatenate([[0.0], np.cumsum(np.full(n - 1, v * dt))]),
            "Speed": np.full(n, v * 3.6),
            "Speed_ms": np.full(n, v),
            "Throttle": np.full(n, 100.0),
            "Brake": np.zeros(n),
            "Gear": np.full(n, 8),
            "X": np.linspace(0, v * duration, n),
            "Y": np.zeros(n),
        }
    )


def test_distance_integration_constant_speed():
    from core.telemetry_resampler import distance_from_speed

    time = np.arange(0, 10, 0.1)
    speed = np.full(len(time), 20.0)

    dist = distance_from_speed(time, speed)

    assert len(dist) == len(time)
    assert dist[0] == 0.0
    assert np.all(np.diff(dist) > 0)
    assert abs(dist[-1] - 20.0 * 9.9) < 0.5


def test_grid_is_monotonic_and_capped():
    from core.telemetry_resampler import common_grid

    dist_a = np.linspace(0, 1000, 50)
    dist_b = np.linspace(0, 950, 40)

    grid = common_grid(dist_a, dist_b)

    assert grid[0] == 0.0
    assert abs(grid[-1] - 950.0) <= 1.0
    assert np.all(np.diff(grid) > 0)
    assert len(grid) <= 20001


def test_resample_nan_channel_finite():
    from core.telemetry_resampler import resample_on_distance

    dist = np.arange(0.0, 91.0, 1.0)
    speed = 10.0 + dist
    profile = pd.DataFrame({"Distance": dist, "Speed": speed, "Speed_ms": speed / 3.6})
    profile.loc[5, "Speed"] = np.nan
    profile.loc[20, "Speed_ms"] = np.nan

    grid = np.arange(0.0, 90.0, 1.0)
    resampled = resample_on_distance(profile, grid)

    assert np.isfinite(resampled["Speed"].to_numpy()).all()
    assert np.isfinite(resampled["Speed_ms"].to_numpy()).all()
    at_15 = resampled.loc[resampled["Distance"] == 15.0, "Speed"].iloc[0]
    assert abs(at_15 - 25.0) < 0.6


def test_delta_time_positive_when_b_slower():
    from core.telemetry_resampler import delta_time

    dist = np.linspace(0.0, 1000.0, 1001)
    time_a = dist / 20.0
    time_b = dist / 19.0

    delta = delta_time(dist.copy(), dist_a=dist, time_a=time_a, dist_b=dist, time_b=time_b)

    assert np.all(delta >= 0)
    assert np.all(np.diff(delta) >= 0)
    assert abs(delta[-1] - (1000.0 / 19.0 - 1000.0 / 20.0)) < 1e-6


def test_align_profiles_same_grid_and_end_delta():
    from core.telemetry_resampler import align_profiles

    pa = const_profile(20.0, duration=50.0)
    pb = const_profile(19.0, duration=60.0)

    out = align_profiles(pa, pb)

    assert len(out["a"]) == len(out["b"])
    assert np.allclose(out["a"]["Distance"].to_numpy(), out["b"]["Distance"].to_numpy())
    assert np.all(out["delta"]["Delta_s"].to_numpy() >= 0)
    expected = out["grid"][-1] * ((1.0 / 19.0) - (1.0 / 20.0))
    assert abs(out["delta"]["Delta_s"].iloc[-1] - expected) < 0.05
