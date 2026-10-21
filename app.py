import datetime
import html

import pandas as pd
import streamlit as st

from components.stint_plots import stint_evolution_figure
from components.tables import corner_table, stint_table
from components.telemetry_plots import telemetry_figure
from components.track_map import track_map_figure
from core import session_manager, telemetry_resampler

st.set_page_config(page_title="Pit Wall Telemetry", layout="wide")

try:
    css = open("style.css", encoding="utf-8").read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except OSError:
    st.warning("style.css missing; falling back to default theme.")

SESSION_LABELS = {"Q": "Qualifying", "R": "Race", "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3"}


@st.cache_data(show_spinner="Loading race calendar…")
def _events(year: int) -> list:
    return session_manager.available_events(year)


@st.cache_resource(show_spinner="Fetching session from F1 live timing (first load can take a minute)…")
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


@st.cache_data(show_spinner="Aligning telemetry on the distance grid…")
def _analysis(year: int, event: str, code: str, driver_a: str, driver_b: str, lap_a: str, lap_b: str) -> dict:
    session = _session(year, event, code)

    def resolve(lap_label: str, driver: str):
        number = session_manager.parse_lap_label(lap_label)
        if number is None:
            return None
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
    pair = [driver_a, driver_b]
    stats = stats[stats["driver"].isin(pair)]
    lap_evolution = tire_model.valid_laps(session)
    lap_evolution = lap_evolution[lap_evolution["Driver"].isin(pair)]

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


def _driver_card(year: int, code: str, row: pd.Series, tag: str) -> None:
    name = row["name"] if pd.notna(row["name"]) and str(row["name"]).strip() else code
    team = row["team"] if pd.notna(row["team"]) else ""
    color = row.get("color", "#8B9097")
    if pd.isna(color):
        color = "#8B9097"
    image = session_manager.driver_image_path(year, str(name))

    with st.container(key=f"driver-card-{tag.lower()}"):
        photo_col, info_col = st.columns([1, 2], gap="small", vertical_alignment="center")
        with photo_col:
            if image:
                st.image(image, width=96, alt=f"Portrait of {name}")
            else:
                st.markdown(
                    f'<div class="pw-driver-photo-missing">{html.escape(code)}</div>',
                    unsafe_allow_html=True,
                )
        with info_col:
            st.markdown(
                f'<div class="pw-driver-info">'
                f'<div class="pw-driver-top">'
                f'<span class="pw-driver-dot" style="background:{html.escape(str(color))};"></span>'
                f'<span class="pw-driver-code">{html.escape(code)}</span>'
                f'<span class="pw-driver-tag">{tag}</span>'
                f'</div>'
                f'<div class="pw-driver-name">{html.escape(str(name))}</div>'
                f'<div class="pw-driver-team">{html.escape(str(team))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


_AUTH_HINT_TOKENS = (
    "datanotloadederror",
    "sessionnotavailableerror",
    "no data for this session",
    "has not been loaded yet",
    "failed to load",
)


def _auth_hint(detail: str) -> None:
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


def _section(title: str):
    st.markdown(f'<div class="pw-section">{title}</div>', unsafe_allow_html=True)


def _main_view(triple: dict):
    year, event, code = triple["year"], triple["event"], triple["code"]
    try:
        candidates = _drivers(year, event, code)
    except Exception as exc:
        _auth_hint(f"{type(exc).__name__}: {exc}")
        st.error(f"Session load failed: {exc}")
        return

    if len(candidates) < 2:
        st.error("This session does not have two drivers with recorded telemetry.")
        return

    try:
        has_timing = session_manager.has_timing_data(_session(year, event, code))
    except Exception as exc:
        _auth_hint(f"{type(exc).__name__}: {exc}")
        st.error(f"Lap timing data is unavailable for this session: {exc}")
        return
    if not has_timing:
        _auth_hint("timing data has not been loaded yet")
        st.error("This session loaded without lap timing data, so no telemetry comparison is possible.")
        return

    key_ctx = f"{year}-{event}-{code}"
    params = st.query_params
    col_a, col_b = st.columns(2, gap="small")
    with col_a:
        driver_a = st.selectbox(
            "Driver A", candidates.index,
            index=_restore_index(params, "drv_a", list(candidates.index)),
            key=f"driver-a-{key_ctx}",
        )
    with col_b:
        driver_b = st.selectbox(
            "Driver B", candidates.index,
            index=_restore_index(params, "drv_b", list(candidates.index), 1),
            key=f"driver-b-{key_ctx}",
        )
    if driver_a == driver_b:
        st.warning("Driver A and B are the same; pick two different drivers to compare.")
        return

    with col_a:
        _driver_card(year, driver_a, candidates.loc[driver_a], "A")
    with col_b:
        _driver_card(year, driver_b, candidates.loc[driver_b], "B")

    try:
        laps_a = _lap_list(year, event, code, driver_a)
        laps_b = _lap_list(year, event, code, driver_b)
    except Exception as exc:
        _auth_hint(f"{type(exc).__name__}: {exc}")
        st.error(f"Lap metadata failed: {exc}")
        return

    if laps_a.empty or laps_b.empty:
        missing = driver_a if laps_a.empty else driver_b
        st.warning(f"{missing} has no timed laps in this session; pick another driver to compare.")
        return

    options_a = ["Fastest (default)"] + [f"Lap {int(n)} · {_format_lap(t)}" for n, t in zip(laps_a["lap"], laps_a["time_s"])]
    options_b = ["Fastest (default)"] + [f"Lap {int(n)} · {_format_lap(t)}" for n, t in zip(laps_b["lap"], laps_b["time_s"])]
    with col_a:
        lap_a = st.selectbox("Lap A", options_a, index=_restore_index(params, "lap_a", options_a), key=f"lap-a-{key_ctx}-{driver_a}")
    with col_b:
        lap_b = st.selectbox("Lap B", options_b, index=_restore_index(params, "lap_b", options_b), key=f"lap-b-{key_ctx}-{driver_b}")

    st.query_params.update(
        year=str(year), gp=event, code=code,
        drv_a=driver_a, drv_b=driver_b, lap_a=lap_a, lap_b=lap_b,
    )

    with st.spinner("Composing analysis…"):
        try:
            data = _analysis(year, event, code, driver_a, driver_b, lap_a, lap_b)
        except Exception as exc:
            _auth_hint(f"{type(exc).__name__}: {exc}")
            st.error(f"Analysis failed on this session: {exc}. Try different drivers, laps, or clear .fastf1_cache if data looks stale.")
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
    st.plotly_chart(telemetry_figure(data["aligned"]), width="stretch", config={"displayModeBar": False})

    map_col, table_col = st.columns([5, 7], gap="small")
    with map_col:
        _section("Track &amp; advantage")
        st.markdown('<div class="pw-note">Segment colors: the faster driver over that mini-sector.</div>', unsafe_allow_html=True)
        st.plotly_chart(
            track_map_figure(data["aligned"], data["corners_df"], data["rotation"], data["microsectors"]),
            width="stretch", config={"displayModeBar": False},
        )
    with table_col:
        _section("Corner breakdown")
        st.markdown('<div class="pw-note">Per corner: minimum speed, braking onset, throttle pick-up and the local time cost.</div>', unsafe_allow_html=True)
        st.plotly_chart(corner_table(data["breakdown"]), width="stretch", config={"displayModeBar": False})

    _section("Tyre stints")
    _left, _right = st.columns([7, 5], gap="small")
    with _left:
        st.plotly_chart(
            stint_evolution_figure(
                data["lap_evolution"], data["stats"],
                {code: row["color"] for code, row in candidates.loc[[driver_a, driver_b]].iterrows()},
            ),
            width="stretch", config={"displayModeBar": False},
        )
    with _right:
        if data["stats"].empty:
            st.info("No stint reached the 3-lap minimum in this session, so there are no fits to tabulate; lap points still render on the left.")
        else:
            st.plotly_chart(stint_table(data["stats"]), width="stretch", config={"displayModeBar": False})
        offsets = data["offsets"]
        offset_text = "  ".join(
            f"{row['Compound']} {row['offset_s']:+.1f}s" for _, row in offsets.iterrows()
        )
        st.caption(f"Compound medians vs fastest compound: {offset_text}")


def _restore_index(params, key: str, options: list, default: int = 0) -> int:
    value = params.get(key)
    if value in options:
        return options.index(value)
    return default


def _param_int(params, key: str, allowed: list):
    try:
        value = int(params.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if value in allowed else None


def _sidebar() -> dict:
    st.sidebar.title("Pit Wall Telemetry")
    st.sidebar.caption("Telemetry comparison console")

    params = st.query_params

    years = list(range(datetime.date.today().year, 2019, -1))
    year = _param_int(params, "year", years)
    if year is None:
        year = years[0]
    year = st.sidebar.selectbox("Season", years, index=years.index(year), key="season-year")
    st.query_params.update(year=str(year))

    triple = {"year": year, "event": None, "code": None}
    try:
        events = _events(year)
    except Exception as exc:
        st.sidebar.error(f"Race calendar unavailable: {exc}")
        return triple
    if not events:
        st.sidebar.warning("No events found for this season yet.")
        return triple

    event = st.sidebar.selectbox(
        "Grand Prix", events, index=_restore_index(params, "gp", events), key=f"gp-{year}",
    )
    codes = list(SESSION_LABELS.keys())
    code_label = st.sidebar.selectbox(
        "Session", codes, format_func=SESSION_LABELS.get,
        index=_restore_index(params, "code", codes), key="session-code",
    )
    st.query_params.update(gp=event, code=code_label)
    triple = {"year": year, "event": event, "code": code_label}
    st.sidebar.markdown(
        '<div class="pw-status">F1 gates this data behind an F1 TV account. Sign in once with '
        '<code>venv\\Scripts\\python -m fastf1 auth f1tv --authenticate</code>; after that streams come from the API or disk cache.</div>',
        unsafe_allow_html=True,
    )
    return triple


triple = _sidebar()
if triple["event"] is None:
    st.info("Select a season, Grand Prix and session from the sidebar to load telemetry.")
    st.stop()

_main_view(triple)
