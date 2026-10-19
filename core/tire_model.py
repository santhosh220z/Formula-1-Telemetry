import numpy as np
import pandas as pd

_MIN_STINT_LAPS = 3


def valid_laps(session) -> pd.DataFrame:
    laps = session.laps
    keep = [
        "Driver",
        "Stint",
        "Compound",
        "TyreLife",
        "LapTime",
        "PitInTime",
        "PitOutTime",
    ]
    clean = laps[[c for c in keep if c in laps.columns]].copy()
    clean = clean[
        clean["LapTime"].notna()
        & clean["PitInTime"].isna()
        & clean["PitOutTime"].isna()
        & clean["Stint"].notna()
        & clean["Compound"].notna()
    ]
    clean["LapTime_s"] = clean["LapTime"].dt.total_seconds()
    return clean


def stint_stats(session) -> pd.DataFrame:
    laps = valid_laps(session)
    rows = []
    for (driver, stint, compound), group in laps.groupby(["Driver", "Stint", "Compound"], observed=True):
        n = len(group)
        if n < _MIN_STINT_LAPS:
            continue
        slope, intercept = np.polyfit(group["TyreLife"], group["LapTime_s"], 1)
        rows.append(
            {
                "driver": driver,
                "stint": int(stint),
                "compound": compound,
                "laps": n,
                "base_s": float(intercept),
                "deg_slope_s_per_lap": float(slope),
                "deg_total_s": float(slope * n),
            }
        )
    return pd.DataFrame(rows)


def compound_offsets(session) -> pd.DataFrame:
    laps = valid_laps(session)
    med = laps.groupby("Compound", observed=True)["LapTime_s"].median()
    base = float(med.min())
    return pd.DataFrame(
        {"Compound": med.index, "median_s": med.to_numpy(), "offset_s": med.to_numpy() - base}
    )


def stint_degradation(session) -> tuple[pd.DataFrame, pd.DataFrame]:
    return stint_stats(session), compound_offsets(session)
