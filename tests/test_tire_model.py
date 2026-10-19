import numpy as np
import pandas as pd
from types import SimpleNamespace


def stint_laps(
    driver: str,
    compound: str,
    intercept: float,
    slope: float,
    tyre_lifes,
    stint: int = 1,
    pit_rows=(),
) -> pd.DataFrame:
    tyre_lifes = list(tyre_lifes)
    rows = []
    for life in tyre_lifes:
        rows.append(
            {
                "Driver": driver,
                "Stint": stint,
                "Compound": compound,
                "TyreLife": life,
                "LapTime": pd.to_timedelta(intercept + slope * life, unit="s"),
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
            }
        )
    for idx in pit_rows:
        rows[idx]["PitInTime"] = pd.to_timedelta(0, unit="s")
    return pd.DataFrame(rows)


def fake_session(laps: pd.DataFrame):
    return SimpleNamespace(laps=laps)


def pit_mix_driver() -> pd.DataFrame:
    good = stint_laps("VER", "SOFT", 90.0, 0.3, range(1, 11))
    garbage = pd.DataFrame(
        [
            {"Driver": "VER", "Stint": 2, "Compound": "SOFT", "TyreLife": 1,
             "LapTime": pd.to_timedelta(50, unit="s"), "PitInTime": pd.NaT, "PitOutTime": pd.NaT},
            {"Driver": "VER", "Stint": 2, "Compound": "SOFT", "TyreLife": 2,
             "LapTime": pd.NaT, "PitInTime": pd.NaT, "PitOutTime": pd.NaT},
        ]
    )
    bad = stint_laps("VER", "SOFT", 999.0, 0.0, [11, 12], stint=2, pit_rows=(0,))
    out = pd.concat([good, garbage, bad], ignore_index=True)
    return fake_session(out)


def test_valid_laps_drops_pit_and_na_laps():
    from core.tire_model import valid_laps

    session = pit_mix_driver()

    laps = valid_laps(session)

    assert len(laps) == 12  # pit in-lap, NaT lap, and the pit-marked lap drop out
    assert "LapTime_s" in laps.columns
    assert laps["PitInTime"].isna().all()
    assert laps["LapTime_s"].notna().all()


def test_stint_stats_recovers_known_slope():
    from core.tire_model import stint_stats

    session = pit_mix_driver()

    stats = stint_stats(session)

    row = stats.iloc[0]
    assert row["driver"] == "VER"
    assert row["compound"] == "SOFT"
    assert row["laps"] == 10
    assert abs(row["deg_slope_s_per_lap"] - 0.3) < 0.01
    assert abs(row["base_s"] - 90.0) < 0.05
    assert abs(row["deg_total_s"] - 3.0) < 0.05


def test_compound_offsets_relative_to_fastest():
    from core.tire_model import compound_offsets

    laps = pd.concat(
        [
            stint_laps("VER", "SOFT", 90.0, 0.5, range(1, 11)),
            stint_laps("VER", "MEDIUM", 92.0, 0.5, range(1, 11)),
        ],
        ignore_index=True,
    )
    session = fake_session(laps)

    offsets = compound_offsets(session)

    soft = offsets[offsets["Compound"] == "SOFT"].iloc[0]
    medium = offsets[offsets["Compound"] == "MEDIUM"].iloc[0]
    assert soft["offset_s"] == 0.0
    assert abs((medium["offset_s"] - soft["offset_s"]) - 2.0) < 0.05


def test_stint_degradation_pair_shape():
    from core.tire_model import stint_degradation

    stats, offsets = stint_degradation(pit_mix_driver())

    assert len(stats) == 1
    assert {"Compound", "median_s", "offset_s"}.issubset(offsets.columns)
