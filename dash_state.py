import datetime

import pandas as pd
import streamlit as st

from core import session_manager

SESSION_LABELS = {"Q": "Qualifying", "R": "Race", "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3"}


def inject_theme() -> None:
    try:
        css = open("style.css", encoding="utf-8").read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except OSError:
        st.warning("style.css missing; falling back to default theme.")


@st.cache_data(show_spinner="Loading race calendar…")
def events(year: int) -> list:
    return session_manager.available_events(year)


@st.cache_resource(show_spinner="Fetching session from F1 live timing (first load can take a minute)…")
def session(year: int, event: str, code: str):
    return session_manager.load_session(year, event, code)


@st.cache_data(show_spinner=False)
def drivers(year: int, event: str, code: str) -> pd.DataFrame:
    return session_manager.driver_candidates(session(year, event, code))


@st.cache_data(show_spinner=False)
def circuit(year: int, event: str, code: str):
    return session_manager.get_circuit_info(session(year, event, code))


@st.cache_data(show_spinner=False)
def lap_list(year: int, event: str, code: str, driver: str) -> pd.DataFrame:
    laps = session_manager.laps_of(session(year, event, code), driver)
    return pd.DataFrame(
        {
            "lap": laps["LapNumber"].to_numpy(dtype=int),
            "time_s": laps["LapTime"].dt.total_seconds().to_numpy(),
        }
    )


_AUTH_HINT_TOKENS = (
    "datanotloadederror",
    "sessionnotavailableerror",
    "no data for this session",
    "has not been loaded yet",
    "failed to load",
)


def auth_hint(detail: str) -> None:
    text = detail.lower()
    if not any(token in text for token in _AUTH_HINT_TOKENS):
        return
    st.info(
        "F1 now requires an F1 TV subscription to serve timing and telemetry data. "
        "Sign in once with your F1 TV account:\n\n"
        "`venv\\Scripts\\python -m fastf1 auth f1tv --authenticate`\n\n"
        "The stored login lasts about four days; after that run the command again. "
        "An active F1 TV Access, Pro or Premium plan is required. Reload the page "
        "after signing in; the first load may still take a minute."
    )


def format_lap(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes}:{seconds % 60:04.1f}"


def restore_index(params, key: str, options: list, default: int = 0) -> int:
    value = params.get(key)
    if value in options:
        return options.index(value)
    return default


def param_int(params, key: str, allowed: list):
    try:
        value = int(params.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if value in allowed else None


def section(title: str):
    st.markdown(f'<div class="pw-section">{title}</div>', unsafe_allow_html=True)


def sidebar() -> dict:
    st.sidebar.title("Pit Wall Telemetry")
    st.sidebar.caption("Telemetry comparison console")

    params = st.query_params

    years = list(range(datetime.date.today().year, 2019, -1))
    year = param_int(params, "year", years)
    if year is None:
        year = years[0]
    year = st.sidebar.selectbox("Season", years, index=years.index(year), key="season-year")
    st.query_params.update(year=str(year))

    triple = {"year": year, "event": None, "code": None}
    try:
        evts = events(year)
    except Exception as exc:
        st.sidebar.error(f"Race calendar unavailable: {exc}")
        return triple
    if not evts:
        st.sidebar.warning("No events found for this season yet.")
        return triple

    event = st.sidebar.selectbox(
        "Grand Prix", evts, index=restore_index(params, "gp", evts), key=f"gp-{year}",
    )
    codes = list(SESSION_LABELS.keys())
    code_label = st.sidebar.selectbox(
        "Session", codes, format_func=SESSION_LABELS.get,
        index=restore_index(params, "code", codes), key="session-code",
    )
    st.query_params.update(gp=event, code=code_label)
    st.sidebar.markdown(
        '<div class="pw-status">F1 gates this data behind an F1 TV account. Sign in once with '
        '<code>venv\\Scripts\\python -m fastf1 auth f1tv --authenticate</code>; after that streams come from the API or disk cache.</div>',
        unsafe_allow_html=True,
    )
    return {"year": year, "event": event, "code": code_label}
