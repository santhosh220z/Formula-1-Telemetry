import pandas as pd
import streamlit as st

from dash_state import (
    SESSION_LABELS,
    auth_hint,
    circuit,
    drivers,
    format_lap,
    inject_theme,
    lap_list,
    param_int,
    restore_index,
    section,
    session,
    sidebar,
)
from components.stint_plots import stint_evolution_figure
from components.tables import corner_table, stint_table
from components.telemetry_plots import telemetry_figure
from components.track_map import track_map_figure
from core import corner_analyzer, session_manager, telemetry_resampler, tire_model

st.set_page_config(page_title="Pit Wall Telemetry", layout="wide")
inject_theme()


@st.cache_data(show_spinner="Aligning telemetry on the distance grid…")
def analysis(year: int, event: str, code: str, driver_a: str, driver_b: str, lap_a: str, lap_b: str) -> dict:
    ses = session(year, event, code)

    def resolve(lap_label: str, driver: str):
        number = session_manager.parse_lap_label(lap_label)
        if number is None:
            return None
        laps = session_manager.laps_of(ses, driver)
        match = laps[laps["LapNumber"] == number]
        if match.empty:
            raise ValueError(f"Lap {number} has no valid time for {driver}")
        return match.iloc[0]

    aligned = telemetry_resampler.build_aligned_telemetry(
        ses, driver_a, driver_b, lap_a=resolve(lap_a, driver_a), lap_b=resolve(lap_b, driver_b)
    )

    corners_df, rotation = circuit(year, event, code)
    candidates = drivers(year, event, code)
    aligned["colors"] = {"a": candidates.loc[driver_a, "color"], "b": candidates.loc[driver_b, "color"]}

    breakdown = corner_analyzer.corner_breakdown(aligned, corners_df)
    microsectors = corner_analyzer.microsector_advantage(aligned)
    stats, offsets = tire_model.stint_degradation(ses)
    stats = stats[stats["driver"].isin([driver_a, driver_b])]
    lap_evolution = tire_model.valid_laps(ses)
    lap_evolution = lap_evolution[lap_evolution["Driver"].isin([driver_a, driver_b])]

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


def _main_view(triple: dict):
    year, event, code = triple["year"], triple["event"], triple["code"]
    try:
        candidates = drivers(year, event, code)
    except Exception as exc:
        auth_hint(f"{type(exc).__name__}: {exc}")
        st.error(f"Session load failed: {exc}")
        return

    if len(candidates) < 2:
        st.error("This session does not have two drivers with recorded telemetry.")
        return

    if not session_manager.has_timing_data(session(year, event, code)):
        auth_hint("timing data has not been loaded yet")
        st.error("This session loaded without lap timing data, so no telemetry comparison is possible.")
        return

    key_ctx = f"{year}-{event}-{code}"
    params = st.query_params
    col_a, col_b = st.columns(2, gap="small")
    with col_a:
        driver_a = st.selectbox(
            "Driver A", candidates.index,
            index=restore_index(params, "drv_a", list(candidates.index)),
            key=f"driver-a-{key_ctx}",
        )
    with col_b:
        driver_b = st.selectbox(
            "Driver B", candidates.index,
            index=restore_index(params, "drv_b", list(candidates.index), 1),
            key=f"driver-b-{key_ctx}",
        )
    if driver_a == driver_b:
        st.warning("Driver A and B are the same; pick two different drivers to compare.")
        return

    try:
        laps_a = lap_list(year, event, code, driver_a)
        laps_b = lap_list(year, event, code, driver_b)
    except Exception as exc:
        auth_hint(f"{type(exc).__name__}: {exc}")
        st.error(f"Lap metadata failed: {exc}")
        return

    if laps_a.empty or laps_b.empty:
        missing = driver_a if laps_a.empty else driver_b
        st.warning(f"{missing} has no timed laps in this session; pick another driver to compare.")
        return

    options_a = ["Fastest (default)"] + [f"Lap {int(n)} · {format_lap(t)}" for n, t in zip(laps_a["lap"], laps_a["time_s"])]
    options_b = ["Fastest (default)"] + [f"Lap {int(n)} · {format_lap(t)}" for n, t in zip(laps_b["lap"], laps_b["time_s"])]
    with col_a:
        lap_a = st.selectbox("Lap A", options_a, index=restore_index(params, "lap_a", options_a), key=f"lap-a-{key_ctx}-{driver_a}")
    with col_b:
        lap_b = st.selectbox("Lap B", options_b, index=restore_index(params, "lap_b", options_b), key=f"lap-b-{key_ctx}-{driver_b}")

    st.query_params.update(
        year=str(year), gp=event, code=code,
        drv_a=driver_a, drv_b=driver_b, lap_a=lap_a, lap_b=lap_b,
    )

    with st.spinner("Composing analysis…"):
        try:
            data = analysis(year, event, code, driver_a, driver_b, lap_a, lap_b)
        except Exception as exc:
            auth_hint(f"{type(exc).__name__}: {exc}")
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

    section("Synced telemetry")
    st.markdown('<div class="pw-note">Zoom or pan any trace; every viewport moves together on the shared distance axis.</div>', unsafe_allow_html=True)
    st.plotly_chart(telemetry_figure(data["aligned"]), use_container_width=True, config={"displayModeBar": False})

    map_col, table_col = st.columns([5, 7], gap="small")
    with map_col:
        section("Track &amp; advantage")
        st.markdown('<div class="pw-note">Segment colors: the faster driver over that mini-sector.</div>', unsafe_allow_html=True)
        st.plotly_chart(
            track_map_figure(data["aligned"], data["corners_df"], data["rotation"], data["microsectors"]),
            use_container_width=True, config={"displayModeBar": False},
        )
    with table_col:
        section("Corner breakdown")
        st.markdown('<div class="pw-note">Per corner: minimum speed, braking onset, throttle pick-up and the local time cost.</div>', unsafe_allow_html=True)
        st.plotly_chart(corner_table(data["breakdown"]), use_container_width=True, config={"displayModeBar": False})

    section("Tyre stints")
    _left, _right = st.columns([7, 5], gap="small")
    with _left:
        st.plotly_chart(
            stint_evolution_figure(
                data["lap_evolution"], data["stats"],
                {code: row["color"] for code, row in candidates.loc[[driver_a, driver_b]].iterrows()},
            ),
            use_container_width=True, config={"displayModeBar": False},
        )
    with _right:
        if data["stats"].empty:
            st.info("No stint reached the 3-lap minimum in this session, so there are no fits to tabulate; lap points still render on the left.")
        else:
            st.plotly_chart(stint_table(data["stats"]), use_container_width=True, config={"displayModeBar": False})
        offsets = data["offsets"]
        offset_text = "  ".join(
            f"{row['Compound']} {row['offset_s']:+.1f}s" for _, row in offsets.iterrows()
        )
        st.caption(f"Compound medians vs fastest compound: {offset_text}")


triple = sidebar()
if triple["event"] is None:
    st.info("Select a season, Grand Prix and session from the sidebar to load telemetry.")
    st.stop()

_main_view(triple)
