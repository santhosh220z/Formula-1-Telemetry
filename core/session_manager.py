import os

import fastf1
import fastf1.plotting
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_REPO_ROOT, ".fastf1_cache")

SESSION_CODES = ["Q", "R", "FP1", "FP2", "FP3"]

_FALLBACK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

_cache_enabled = False


def enable_cache() -> None:
    global _cache_enabled
    if _cache_enabled:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    _cache_enabled = True


def validate_session_code(code: str) -> str:
    session_code = str(code).upper()
    if session_code not in SESSION_CODES:
        raise ValueError(f"Unsupported session code {code!r}; choose from {SESSION_CODES}")
    return session_code


def filter_event_schedule(schedule: pd.DataFrame) -> list[str]:
    names = schedule["EventName"].dropna()
    is_testing = names.str.lower().str.contains("test")
    return list(names[~is_testing])


def available_events(year: int) -> list[str]:
    return filter_event_schedule(fastf1.get_event_schedule(year))


def load_session(year: int, event: str, code: str):
    enable_cache()
    session = fastf1.get_session(year, event, validate_session_code(code))
    session.load()
    return session


def driver_candidates(session) -> pd.DataFrame:
    laps = session.laps
    rows = []
    for code, group in laps.groupby("Driver"):
        first = group.iloc[0]
        rows.append(
            {"code": code, "name": first.get("FullName", code), "team": first.get("TeamName", "")}
        )
    candidates = pd.DataFrame(rows).set_index("code").sort_index()
    colors = []
    for i, code in enumerate(candidates.index):
        try:
            colors.append(fastf1.plotting.get_driver_color(code, session))
        except Exception:
            colors.append(_FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
    candidates["color"] = colors
    return candidates[["name", "team", "color"]]


def get_circuit_info(session):
    circuit_info = session.get_circuit_info()
    return circuit_info.corners, circuit_info.rotation


def has_timing_data(session) -> bool:
    laps = session.laps
    if "LapTime" not in laps.columns or laps.empty:
        return False
    return bool(laps["LapTime"].notna().any())


def best_lap(session, driver: str):
    driver_laps = session.laps.pick_drivers(driver)
    lap = driver_laps.pick_fastest()
    if lap is None or pd.isna(getattr(lap, "LapTime", None)):
        raise ValueError(f"No laps with a valid LapTime for driver {driver!r}")
    return lap


def laps_of(session, driver: str) -> pd.DataFrame:
    driver_laps = session.laps.pick_drivers(driver)
    timed = driver_laps[driver_laps["LapTime"].notna()]
    return timed.sort_values("LapNumber")
