import numpy as np
import pandas as pd


def _lap_profile(lap_no: int, t0: float, duration: float = 60.0, n: int = 8, speed: float = 30.0):
    t = t0 + np.linspace(0.0, duration, n)
    return (
        lap_no,
        pd.DataFrame(
            {
                "Time": t,
                "X": np.linspace(0.0, speed * duration, n),
                "Y": np.full(n, float(lap_no)),
                "Speed": np.full(n, speed * 3.6),
            }
        ),
    )


def _curves(laps):
    from core.lap_replay import session_curves

    return session_curves(laps)


def test_curves_concatenate_and_dedupe():
    lap_one = _lap_profile(1, 0.0, 60.0)
    lap_two = _lap_profile(2, 60.0, 60.0)  # first sample overlaps lap one's last
    curve = _curves([lap_one, lap_two])

    t = curve["t"]
    assert np.all(np.diff(t) > 0)
    assert len(t) == 15  # 8 + 8 minus the duplicate seam sample


def test_curve_windows_track_lap_bounds():
    lap_one = _lap_profile(1, 10.0, 60.0)
    lap_two = _lap_profile(2, 120.0, 45.0)
    curve = _curves([lap_one, lap_two])

    assert np.allclose(curve["lap_starts"], [10.0, 120.0])
    assert np.allclose(curve["lap_ends"], [70.0, 165.0])
    assert curve["lap_numbers"] == [1, 2]


def test_position_at_window_start_is_lap_start():
    lap_one = _lap_profile(1, 10.0, 60.0)
    curve = _curves([lap_one])

    from core.lap_replay import replay_frames

    frames = replay_frames({"VER": curve}, n_frames=3)
    fr = frames["drivers"]["VER"]
    assert np.isclose(fr["x"][0], 0.0)
    assert np.isclose(fr["x"][-1], 30.0 * 60.0)


def test_position_outside_windows_is_nan():
    lap_one = _lap_profile(1, 10.0, 60.0)
    lap_two = _lap_profile(2, 120.0, 60.0)
    curve = _curves([lap_one, lap_two])

    from core.lap_replay import replay_frames

    frames = replay_frames({"VER": curve}, n_frames=11)
    fr = frames["drivers"]["VER"]
    gap = (frames["t"] > 70.0) & (frames["t"] < 120.0)
    assert np.isnan(fr["x"][gap]).all()
    assert np.isfinite(fr["x"][~gap]).all()


def test_speed_nan_outside_window():
    lap_one = _lap_profile(1, 10.0, 60.0)
    lap_two = _lap_profile(2, 120.0, 60.0)
    curve = _curves([lap_one, lap_two])

    from core.lap_replay import replay_frames

    frames = replay_frames({"VER": curve}, n_frames=11)
    fr = frames["drivers"]["VER"]
    gap = (frames["t"] > 70.0) & (frames["t"] < 120.0)
    assert np.isnan(fr["speed"][gap]).all()
    assert np.isclose(fr["speed"][frames["t"] <= 26.0][0], 30.0 * 3.6)


def test_replay_lap_index_maps_visible_frames():
    lap_one = _lap_profile(1, 10.0, 60.0)
    lap_two = _lap_profile(2, 120.0, 60.0)
    curve = _curves([lap_one, lap_two])

    from core.lap_replay import replay_frames

    frames = replay_frames({"VER": curve}, n_frames=21)
    fr = frames["drivers"]["VER"]
    mid = frames["t"]
    inside_one = (mid >= 10.0) & (mid <= 70.0)
    inside_two = (mid >= 120.0) & (mid <= 180.0)
    assert (fr["lap"][inside_one] == 1).all()
    assert (fr["lap"][inside_two] == 2).all()
    assert (fr["lap"][~(inside_one | inside_two)] == -1).all()


def test_crop_curve_removes_outside_samples():
    lap_one = _lap_profile(1, 10.0, 60.0)
    lap_two = _lap_profile(2, 120.0, 60.0)
    curve = _curves([lap_one, lap_two])

    from core.lap_replay import crop_curve

    cropped = crop_curve(curve, 125.0, 150.0)
    assert cropped["t"].min() >= 125.0
    assert cropped["t"].max() <= 150.0
    assert cropped["lap_numbers"] == [2]


def test_span_covers_all_drivers():
    a = _curves([_lap_profile(1, 10.0, 60.0)])
    b = _curves([_lap_profile(1, 500.0, 90.0)])

    from core.lap_replay import span_of

    t0, t1 = span_of({"VER": a, "VET": b})
    assert np.isclose(t0, 10.0)
    assert np.isclose(t1, 590.0)


def test_frame_count_auto_bounds():
    short = _curves([_lap_profile(1, 0.0, 30.0)])
    long = _curves([_lap_profile(i, i * 100.0, 60.0) for i in range(1, 60)])

    from core.lap_replay import replay_frames

    assert replay_frames({"VER": short})["n_frames"] >= 90
    assert replay_frames({"VER": long})["n_frames"] <= 600


def test_position_progresses_monotonically_inside_lap():
    lap_one = _lap_profile(1, 0.0, 60.0, speed=30.0)
    curve = _curves([lap_one])

    from core.lap_replay import replay_frames

    frames = replay_frames({"VER": curve}, n_frames=7)
    fr = frames["drivers"]["VER"]
    xs = fr["x"][:5]  # all frames inside the single lap window
    assert np.all(np.diff(xs) > 0)
