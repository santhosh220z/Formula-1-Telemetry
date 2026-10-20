import numpy as np
import pandas as pd


def corner_profile(v_straight: float, v_corner: float) -> pd.DataFrame:
    """Straight 0-150 m, brake ramp 150-200 m, plateau 200-300 m, throttle+accel 300-350 m."""
    d = np.arange(0.0, 1000.0, 1.0)
    speed = np.empty_like(d)
    speed[:] = v_straight
    ramp1 = (d >= 150.0) & (d < 200.0)
    speed[ramp1] = v_straight + (v_corner - v_straight) * (d[ramp1] - 150.0) / 50.0
    plateau = (d >= 200.0) & (d < 300.0)
    speed[plateau] = v_corner
    ramp2 = (d >= 300.0) & (d < 350.0)
    speed[ramp2] = v_corner + (v_straight - v_corner) * (d[ramp2] - 300.0) / 50.0

    step_avg = 0.5 * (speed[:-1] + speed[1:])
    dt = 1.0 / step_avg
    time = np.concatenate([[0.0], np.cumsum(dt)])

    throttle = np.where((d >= 150.0) & (d < 310.0), 0.0, 95.0)
    brake = np.where((d >= 150.0) & (d < 205.0), 100.0, 0.0)

    return pd.DataFrame(
        {
            "Time": time,
            "Distance": d,
            "Speed": speed * 3.6,
            "Speed_ms": speed,
            "Throttle": throttle,
            "Brake": brake,
            "Gear": np.full(len(d), 8),
            "X": d,
            "Y": np.zeros(len(d)),
        }
    )


def align() -> dict:
    from core.telemetry_resampler import align_profiles

    pa = corner_profile(20.0, 20.0)
    pb = corner_profile(21.0, 17.0)
    return align_profiles(pa, pb)


def locate_corners(aligned: dict, corners: pd.DataFrame) -> pd.DataFrame:
    from core.corner_analyzer import locate_corners as locate

    return locate(aligned, corners)


def one_corner_df() -> pd.DataFrame:
    return pd.DataFrame({"Number": [1], "Letter": [""], "X": [200.0], "Y": [0.0]})


def test_locate_corners_projects_marker_to_distance():
    aligned = align()

    located = locate_corners(aligned, one_corner_df())

    assert located.iloc[0]["Distance"] == 200.0
    assert located.iloc[0]["Number"] == 1


def test_corner_metrics_brake_onset_and_throttle_pickup():
    from core.corner_analyzer import corner_window_metrics

    aligned = align()

    m = corner_window_metrics(aligned, 200.0)

    assert abs(m["a_v_min_ms"] - 20.0) < 0.2
    assert abs(m["b_v_min_ms"] - 17.0) < 0.2
    assert m["b_brake_onset_dist"] >= 150.0  # braking begins at the 150 m ramp start
    assert 150.0 <= m["b_brake_onset_dist"] <= 175.0
    assert m["b_throttle_pickup_dist"] > 308.9  # sustained throttle begins at the 310 m ramp
    assert 305.0 <= m["b_throttle_pickup_dist"] <= 330.0


def test_corner_breakdown_reports_faster_driver_in_corner():
    from core.corner_analyzer import corner_breakdown

    aligned = align()

    breakdown = corner_breakdown(aligned, one_corner_df())

    row = breakdown.iloc[0]
    assert row["corner_delta_s"] > 0.5
    assert row["faster"] == "A"
    assert abs(row["b_v_min_ms"] - 17.0) < 0.2


def test_ties_give_no_faster_driver():
    from core.telemetry_resampler import align_profiles
    from core.corner_analyzer import corner_breakdown

    pa = corner_profile(20.0, 20.0)
    identical = align_profiles(pa, pa.copy())

    breakdown = corner_breakdown(identical, one_corner_df())

    assert abs(breakdown.iloc[0]["corner_delta_s"]) < 1e-6
    assert breakdown.iloc[0]["faster"] == "Tie"


def test_microsector_advantage_selects_faster_segments():
    from core.corner_analyzer import microsector_advantage

    aligned = align()

    micro = microsector_advantage(aligned, n=5)

    assert list(micro["faster"]) == ["B", "A", "B", "B", "B"]
    assert np.all(micro.loc[micro["faster"] == "A", "adv_s"].to_numpy() > 0)
    assert np.all(micro.loc[micro["faster"] == "B", "adv_s"].to_numpy() < 0)
