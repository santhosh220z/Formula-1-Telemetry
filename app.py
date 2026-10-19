import datetime

import pandas as pd
import streamlit as st

from components.stint_plots import stint_evolution_figure
from components.tables import corner_table, stint_table
from components.telemetry_plots import telemetry_figure
from components.track_map import track_map_figure
from core import session_manager, telemetry_resampler

st.set_page_config(page_title="Pit Wall Telemetry", layout="wide")

try:
    st.markdown(open("style.css", encoding="utf-8").read(), unsafe_allow_html=True)
except OSError:
    st.warning("style.css missing; falling back to default theme.")

SESSION_LABELS = {"Q": "Qualifying", "R": "Race", "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3"}


@st.cache_data(show_spinner="Loading race calendar")
def _events(year: int) -> list:
    return session_manager.available_events(year)


@st.cache_resource(show_spinner="Fetching session from F1 live timing (first load can take a minute)")
def _session(year: int, event: str, code: str):
    return session_manager.load_session(year, event, code)


@st.cache_data(show_spinner=False)
def _drivers(year: int, event: str, code: str) -> pd.DataFrame:
    return session_manager.driver_candidates(_session(year, event, code))


@st.cache_data(show_spinner=False)
def _circuit(year: int, event: str, code: str):
    return session_manager.get_circuit_info(_session(year, event, code))


@st.cache_data(show_spinner=False)
def _lap_list(year: int, event: str, code: str, driver: str) -> pd.DataFrame:
    laps = session_manager.laps_of(_session(year, event, code), driver)
    return pd.DataFrame(
        {
            "lap": laps["LapNumber"].to_numpy(dtype=int),
            "time_s": laps["LapTime"].dt.total_seconds().to_numpy(),
        }
    )


@st.cache_data(show_spinner="Aligning telemetry on the distance grid")
def _analysis(year: int, event: str, code: str, driver_a: str, driver_b: str, lap_a: str, lap_b: str) -> dict:
    session = _session(year, event, code)

    def resolve(lap_label: str, driver: str):
        if lap_label.startswith("Fastest"):
            return None
        number = int(lap_label.split()[-1])
        laps = session_manager.laps_of(session, driver)
        match = laps[laps["LapNumber"] == number]
        if match.empty:
            raise ValueError(f"Lap {number} has no valid time for {driver}")
        return match.iloc[0]

    aligned = telemetry_resampler.build_aligned_telemetry(
        session, driver_a, driver_b, lap_a=resolve(lap_a, driver_a), lap_b=resolve(lap_b, driver_b)
    )

    from core import corner_analyzer, tire_model

    corners_df, rotation = _circuit(year, event, code)
    candidates = _drivers(year, event, code)
    aligned["colors"] = {"a": candidates.loc[driver_a, "color"], "b": candidates.loc[driver_b, "color"]}

    breakdown = corner_analyzer.corner_breakdown(aligned, corners_df)
    microsectors = corner_analyzer.microsector_advantage(aligned)
    stats, offsets = tire_model.stint_degradation(session)
    lap_evolution = tire_model.valid_laps(session)

    return {
        "aligned": aligned,
        "corners_df": corners_df,
        "rotation": float(rotation) if rotation is not None else 0.0,
        "breakdown": breakdown,
        "microsectors": microsectors,
        "stats": stats,
        "offsets": offsets,
        "lap_evolution": lap_evolution,
    }


def _format_lap(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes}:{seconds % 60:04.1f}"


def _section(title: str):
    st.markdown(f'<div class="pw-section">{title}</div>', unsafe_allow_html=True)


def _main_view(triple: dict):
    year, event, code = triple["year"], triple["event"], triple["code"]
    try:
        candidates = _drivers(year, event, code)
    except Exception as exc:
        st.error(f"Session loaded but driver metadata failed: {exc}")
        return

    if len(candidates) < 2:
        st.error("This session does not have two drivers with recorded telemetry.")
        return

    driver_a = st.selectbox("Driver A", candidates.index, key="driver_a")
    driver_b = st.selectbox(
        "Driver B", candidates.index,
        index=1 if len(candidates) > 1 else 0, key="driver_b",
    )
    if driver_a == driver_b:
        st.warning("Driver A and B are the same — pick two different drivers to compare.")
        return

    try:
        laps_a = _lap_list(year, event, code, driver_a)
        laps_b = _lap_list(year, event, code, driver_b)
        lap_a = st.selectbox("Lap A", ["Fastest (default)"] + [f"Lap {int(n)} — {_format_lap(t)}" for n, t in zip(laps_a["lap"], laps_a["time_s"])], key="lap_a")
        lap_b = st.selectbox("Lap B", ["Fastest (default)"] + [f"Lap {int(n)} — {_format_lap(t)}" for n, t in zip(laps_b["lap"], laps_b["time_s"])], key="lap_b")

        with st.spinner("Composing analysis ..."):
            try:
                data = _analysis(year, event, code, driver_a, driver_b, lap_a, lap_b)
            except Exception as exc:
                st.error(f"Analysis failed on this session: {exc}. Try different drivers, laps, or clear .fastf1_cache if data looks stale.")
                return
    except Exception as exc:
        st.error(f"Lap metadata failed: {exc}")
        return

    colors = data["aligned"].get("colors", {"a": "#8B9097", "b": "#8B9097"})
    head = f"""
    <div class="pw-head">
      <div class="pw-event">{event} &middot; {SESSION_LABELS.get(code, code)} {year}</div>
      <div class="pw-meta">delta = B minus A &middot; positive means driver A faster at that point in the lap</div>
    </div>
    <div style="display:flex; gap:10px; margin:10px 0 2px 0;">
      <span class="pw-chip"><span class="pw-dot" style="background:{colors['a']};"></span>{driver_a}<span class="pw-tag">A</span></span>
      <span class="pw-chip"><span class="pw-dot" style="background:{colors['b']};"></span>{driver_b}<span class="pw-tag">B</span></span>
    </div>
    """
    st.markdown(head, unsafe_allow_html=True)

    _section("Synced telemetry")
    st.markdown('<div class="pw-note">Zoom or pan any trace; every viewport moves together on the shared distance axis.</div>', unsafe_allow_html=True)
    st.plotly_chart(telemetry_figure(data["aligned"]), use_container_width=True, config={"displayModeBar": False})

    map_col, table_col = st.columns([5, 7], gap="small")
    with map_col:
        _section("Track &amp; advantage")
        st.markdown('<div class="pw-note">Segment colors: the faster driver over that mini-sector.</div>', unsafe_allow_html=True)
        st.plotly_chart(
            track_map_figure(data["aligned"], data["corners_df"], data["rotation"], data["microsectors"]),
            use_container_width=True, config={"displayModeBar": False},
        )
    with table_col:
        _section("Corner breakdown")
        st.markdown('<div class="pw-note">Per corner: minimum speed, braking onset, throttle pick-up and the local time cost.</div>', unsafe_allow_html=True)
        st.plotly_chart(corner_table(data["breakdown"]), use_container_width=True, config={"displayModeBar": False})

    _section("Tyre stints")
    _left, _right = st.columns([7, 5], gap="small")
    with _left:
        st.plotly_chart(
            stint_evolution_figure(
                data["lap_evolution"], data["stats"],
                {code: row["color"] for code, row in candidates.iterrows()},
            ),
            use_container_width=True, config={"displayModeBar": False},
        )
    with _right:
        st.plotly_chart(stint_table(data["stats"]), use_container_width=True, config={"displayModeBar": False})
        offsets = data["offsets"]
        offset_text = "  ".join(
            f"{row['Compound']} {row['offset_s']:+.1f}s" for _, row in offsets.iterrows()
        )
        st.caption(f"Compound medians vs fastest compound: {offset_text}")


def _sidebar() -> dict:
    st.sidebar.title("Pit Wall Telemetry")
    st.sidebar.caption("Telemetry comparison console")

    years = list(range(datetime.date.today().year, 2019, -1))
    year = st.sidebar.selectbox("Season", years, index=0)

    triple = {"year": year, "event": None, "code": None}
    try:
        events = _events(year)
    except Exception as exc:
        st.sidebar.error(f"Race calendar unavailable: {exc}")
        return triple
    if not events:
        st.sidebar.warning("No events found for this season yet.")
        return triple

    event = st.sidebar.selectbox("Grand Prix", events)
    code_label = st.sidebar.selectbox("Session", list(SESSION_LABELS.keys()), format_func=SESSION_LABELS.get)
    triple = {"year": year, "event": event, "code": code_label}
    st.sidebar.markdown(
        '<div class="pw-status">First load of a session fetches from the F1 API; later loads come from disk cache.</div>',
        unsafe_allow_html=True,
    )
    return triple


triple = _sidebar()
if triple["event"] is None:
    st.info("Select a season, Grand Prix and session from the sidebar to load telemetry.")
    st.stop()

_main_view(triple)
