import streamlit as st

from dash_state import (
    SESSION_LABELS,
    auth_hint,
    circuit,
    drivers,
    format_lap,
    inject_theme,
    lap_list,
    restore_index,
    section,
    session,
    sidebar,
)
from components.lap_replay import replay_figure
from core import lap_replay, session_manager
from core.telemetry_resampler import extract_lap_profile

st.set_page_config(page_title="Lap Replay · Pit Wall Telemetry", layout="wide")
inject_theme()

_PACES = {"15×": 15.0, "30×": 30.0, "60×": 60.0, "120×": 120.0}
_MAX_DRIVERS = 6


@st.cache_data(show_spinner="Building replay curves…")
def curves(year: int, event: str, code: str, driver_list: tuple) -> dict:
    ses = session(year, event, code)
    out = {}
    for drv in driver_list:
        laps = session_manager.laps_of(ses, drv)
        pairs = [
            (int(row["LapNumber"]), extract_lap_profile(row))
            for _, row in laps.iterrows()
        ]
        out[drv] = lap_replay.session_curves(pairs)
    return out


def _restore_drivers(params) -> list:
    value = params.get("r_drivers")
    if not value:
        return []
    return value.split(",")


triple = sidebar()
if triple["event"] is None:
    st.info("Select a season, Grand Prix and session from the sidebar to load telemetry.")
    st.stop()

year, event, code = triple["year"], triple["event"], triple["code"]
try:
    candidates = drivers(year, event, code)
except Exception as exc:
    auth_hint(f"{type(exc).__name__}: {exc}")
    st.error(f"Session load failed: {exc}")
    st.stop()

if not session_manager.has_timing_data(session(year, event, code)):
    auth_hint("timing data has not been loaded yet")
    st.error("This session loaded without lap timing data, so nothing can be replayed.")
    st.stop()

key_ctx = f"{year}-{event}-{code}"
params = st.query_params

mode = st.radio(
    "What to replay",
    ["All laps (session playback)", "One lap"],
    index=restore_index(params, "r_mode", ["One lap", "All laps (session playback)"]),
    key=f"replay-mode-{key_ctx}",
    horizontal=True,
)
st.query_params.update(r_mode=mode)

col_mode, col_pace = st.columns([3, 1], gap="small")
with col_mode:
    if mode == "All laps (session playback)":
        default = _restore_drivers(params) or list(candidates.index[:2])
        default = [d for d in default if d in candidates.index][: _MAX_DRIVERS] or list(candidates.index[:1])
        selected = st.multiselect(
            f"Drivers on track (up to {_MAX_DRIVERS})",
            candidates.index.tolist(),
            default=default,
            max_selections=_MAX_DRIVERS,
            key=f"replay-drivers-{key_ctx}",
        )
        st.query_params.update(r_drivers=",".join(selected))
    else:
        driver = st.selectbox(
            "Driver", candidates.index,
            index=restore_index(params, "r_drv", list(candidates.index)),
            key=f"replay-driver-{key_ctx}",
        )
        st.query_params.update(r_drv=driver)
        try:
            laps = lap_list(year, event, code, driver)
            lap_labels = [f"Lap {int(n)} · {format_lap(t)}" for n, t in zip(laps["lap"], laps["time_s"])]
            label = st.selectbox(
                "Lap", lap_labels,
                index=restore_index(params, "r_lap", lap_labels),
                key=f"replay-lap-{key_ctx}-{driver}",
            )
            st.query_params.update(r_lap=label)
        except Exception as exc:
            auth_hint(f"{type(exc).__name__}: {exc}")
            st.error(f"Lap metadata failed: {exc}")
            st.stop()
        selected = [driver]

with col_pace:
    pace_label = st.select_slider(
        "Playback speed", list(_PACES), value="60×",
        key=f"replay-pace-{key_ctx}",
    )
st.query_params.update(r_pace=pace_label)

if not selected:
    st.info("Pick at least one driver to replay their laps against the session clock.")
    st.stop()

try:
    curve_map = curves(year, event, code, tuple(selected))
except Exception as exc:
    auth_hint(f"{type(exc).__name__}: {exc}")
    st.error(f"Replay data failed: {exc}. Try different drivers or clear .fastf1_cache if data looks stale.")
    st.stop()

replay_curves = {}
caption_bits = []
if mode == "One lap":
    driver = selected[0]
    curve = curve_map[driver]
    number = int(label.split(" · ")[0].replace("Lap ", ""))
    if number in curve["lap_numbers"]:
        idx = curve["lap_numbers"].index(number)
        replay_curves[driver] = lap_replay.crop_curve(curve, curve["lap_starts"][idx], curve["lap_ends"][idx])
        caption_bits.append(f"{driver} Lap {number} · {format_lap(float(curve['lap_ends'][idx] - curve['lap_starts'][idx]))}")
    else:
        st.warning("That lap has no position telemetry recorded.")
        st.stop()
else:
    replay_curves.update(curve_map)

try:
    frames = lap_replay.replay_frames(replay_curves)
except Exception as exc:
    st.error(f"Playback layout failed: {exc}")
    st.stop()

corners_df, rotation = circuit(year, event, code)
colors = {name: candidates.loc[name, "color"] for name in selected}

head = f"""
<div class="pw-head">
  <div class="pw-event">{event} &middot; {SESSION_LABELS.get(code, code)} {year}</div>
  <div class="pw-meta">session-clock playback &middot; dots follow each timed lap at its recorded pace</div>
</div>
<div style="display:flex; gap:10px; margin:10px 0 2px 0;">
  {''.join(f'<span class="pw-chip"><span class="pw-dot" style="background:{colors[name]};"></span>{name}</span>' for name in selected)}
</div>
"""
st.markdown(head, unsafe_allow_html=True)

section("Lap replay")
st.markdown(
    '<div class="pw-note">Real session timing: dots appear only while their lap telemetry is live, hiding during out-laps and pit stops. Slider scrubs; Play resumes from the handle.</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(
    replay_figure(frames, replay_curves, colors, corners_df, rotation, _PACES[pace_label]),
    use_container_width=True,
    config={"displayModeBar": False},
)

bits = "  ·  ".join(caption_bits) if caption_bits else f"{len(selected)} drivers replaying every timed lap"
st.caption(bits)
