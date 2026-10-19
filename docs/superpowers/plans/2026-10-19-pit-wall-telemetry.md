# Pit Wall Telemetry Implementation Plan

> **For agentic workers:** Native execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit dashboard comparing two F1 drivers' telemetry per (2020+ season, GP, session), with synced 5-trace plots, track map + corner apexes + mini-sector advantage, corner metrics, and stint tire degradation.

**Architecture:** `core/` (pure fastf1/numpy/pandas/scipy, no streamlit) computes aligned telemetry on a common distance grid; `components/` builds Plotly figures; `app.py` wires Streamlit widgets + caches only.

**Tech Stack:** Python 3.10+ (dev box has 3.13), fastf1 (pinned), streamlit, plotly (GO + make_subplots only), pandas, numpy, scipy, pytest.

**Spec:** `AGENTS.md` "Product contract" section (spec of record).

## Global Constraints

- Python 3.10+; deps pinned `==` in requirements.txt; no torch/sklearn/matplotlib.
- Plotly graph objects only; no `st.line_chart`/`st.pyplot`.
- `core/` imports fastf1 but never streamlit; `app.py`/`components/` never import fastf1.
- FastF1 cache: `.fastf1_cache/`, enabled before any `session.load()`; `st.cache_resource` (resource: object) for Session; `st.cache_data` for frames.
- Never hardcode GP-name maps; dropdowns from `fastf1.get_event_schedule(year)`.
- Session codes: `Q`, `R`, `FP1`–`FP3` only (v1). Seasons 2020–current.
- Lap comparison defaults: fastest lap per driver; optional lap pickers in sidebar.
- Tire model: per-stint linear fit (lap time vs TyreLife) + per-compound median offsets; no ML.
- Dev box: Windows/PowerShell; `venv\Scripts\` prefixes.
- One commit per task; TDD for every `core/` module (test seen failing first).

## Review Focus

1. Rain/partial sessions with sparse or missing laps → app shows a defined error state, no traceback (task 8 catches; unit: `core` functions raise typed `ValueError` on empty inputs, tasks 2–4).
2. Driver lap missing telemetry channels (Speed/Throttle/Brake/DRS gaps) → resampler fills outliers, does not crash (task 2 test: NaN channel channel).
3. WET/INTERMEDIATE compounds mixed with slicks → compound offset normalization excludes sparse compounds gracefully (task 4 test).
4. Tight tracks (Monaco) & GPS jitter → distances from 1 m-adaptive grid; non-monotonic pos data destabilizes delta math (task 2 test).
5. First-run Windows cache dir auto-creation + slow first load → loading state in app; cache dir created in `session_manager` (task 1/8).

---

### Task 0: Scaffold

- [x-plan] `requirements.txt` pinned: `fastf1`, `streamlit`, `plotly`, `pandas`, `numpy`, `scipy`, `pytest` (exact versions from pip freeze).
- [x-plan] `.gitignore`: `venv/`, `__pycache__/`, `.fastf1_cache/`, `*.pyc`.
- [x-plan] `core/__init__.py`, `components/__init__.py` empty. README stub.

### Task 1: core/session_manager.py

**Files:** create `core/session_manager.py`, `tests/test_session_manager.py`

**Interfaces:**
- `validate_session_code(code: str) -> str` (pure, raises ValueError off the allowlist).
- `filter_event_schedule(schedule: pd.DataFrame) -> list[str]` (pure: drop testing events, return official EventName sorted chronologically).
- `available_events(year: int) -> list[str]` (wraps fastf1.get_event_schedule + filter).
- `enable_cache() -> None` (idempotent, points at `.fastf1_cache/`, creates dir).
- `load_session(year: int, event: str, code: str)` → fastf1 Session (calls enable_cache; app wraps in `st.cache_resource`).
- `driver_candidates(session) -> pd.DataFrame` index `code`, cols `[name, team, color]` — color from fastf1.plotting with try/except fallback to a fixed palette.
- `get_circuit_info(session)` → (corners: pd.DataFrame[Number/Letter/X/Y/Distance(±)], rotation: float). Distance located by projecting X/Y onto the fastest-lap path.
- `LAP_PICKER helpers:` `best_lap(session, driver: str) -> fastf1.Lap`, `laps_of(session, driver: str) -> list[fastf1.Lap]`.

Tests (pure only): `validate_session_code` rejects `'X'`, accepts `'Q'`; `filter_event_schedule` drops `Testing` events from a fixture DataFrame; `driver_candidates` on a duck-typed Session fixture (`.laps` DataFrame) returns stable sorted codes; fallback color path works when fastf1.plotting raises.

### Task 2: core/telemetry_resampler.py (delta-time math)

**Files:** create `core/telemetry_resampler.py`, `tests/test_telemetry_resampler.py`

**Interfaces:**
- `lap_distance_profile(session, driver: str, lap=None) -> pd.DataFrame` cols `[Time, Distance, Speed, Throttle, Brake, Gear, X, Y]`. Distance = trapz ∫ v dt (monotonic by construction; fastf1 delta-time approach).
- `common_grid(dist_a: np.ndarray, dist_b: np.ndarray, step: float = 1.0) -> np.ndarray` — clipped to both signals, capped at 10_000 m ≤ 1 m granularity step-adaptive (`np.linspace(min, max, min(20000, ceil(span/step)))`).
- `resample_on_distance(profile, grid) -> pd.DataFrame` — scipy `interp1d` (linear, `fill_value="extrapolate")` limited to grid ends; NaN-safe.
- `delta_time(grid, t_a: pd.Series/np.ndarray, t_b) -> np.ndarray` — via inverted time-at-distance interpolation; positive = driver b slower at distance d.
- `build_aligned_telemetry(session, driver_a: str, driver_b: str, lap_a=None, lap_b=None) -> dict` keys: `{"a": df, "b": df, "delta": df, "grid": arr, "driver_a", "driver_b", "t_micro"}`. `lap_a/lab_b=None` → fastest lap.

Tests (all synthetic, no network): constant-speed 90 s lap → `bottom-line Distance == 89.9*v` ±tol; two constant-speed drivers v_a=60, v_b=59 → single-peak delta max ≈ (L/v_b − L/v_a) sign positive; NaN channel → resample yields finite except grid ends; grid monotonic & sorted; aligned `a`/`b` frames have identical grids (lengths equal).

### Task 3: core/corner_analyzer.py

**Files:** create `core/corner_analyzer.py`, `tests/test_corner_analyzer.py`

**Interfaces:**
- `locate_corners(aligned: dict, corners: pd.DataFrame) -> pd.DataFrame` — nearest-X/Y projection of corner markers onto a driver's path → `Distance`, keeps Number/Letter.
- `corner_window_metrics(aligned: dict, corner_distance: float, window: tuple[float,float]) -> dict` — per driver: `v_min`, `v_min_dist`, `brake_onset_dist` (sustained speed-gradient < −2 m/s² before v_min; sustained = ≥5 points), `throttle_pickup_dist` (first sustained Throttle > 40% after v_min), `sectors via window micro-splits`.
- `corner_breakdown(aligned, corners, window=(−60.0, 40.0) m) -> pd.DataFrame` — per corner row: corner label, per driver v_min/brake_onset/throttle_pickup, `corner_delta_s` (time difference of the window), faster driver.
- `microsector_advantage(aligned, n: int = 10) -> pd.DataFrame` — n equal-distance segments: faster driver + Δs + segment start/end distance.

Tests (synthetic): triangular speed profile with a V at known distance → `v_min` == trough within 1 m; brake onset strictly before trough distance; throttle pickup strictly after; microsector advantage picks the slower/known-faster driver upstream/downstream; corner_delta sign flips.

### Task 4: core/tire_model.py

**Files:** create `core/tire_model.py`, `tests/test_tire_model.py`

**Interfaces:**
- `valid_laps(session) -> pd.DataFrame` — LapTime notna, no PitInTime/PitOutTime, LapNumber ≥ 1, LapTime_s = total_seconds.
- `stint_stats(session) -> pd.DataFrame` per (Driver, Stint, Compound): `laps`, `base_s` (intercept at TyreLife=0), `deg_slope_s_per_lap`, `deg_total_s` = slope·n_laps; slope from `np.polyfit(deg=1)`.
- `compound_offsets(session) -> pd.DataFrame` per compound: median LapTime offset vs fastest median compound slick baseline.
- `stint_degradation(session) -> (stats: pd.DataFrame, summary: pd.DataFrame)` — summary per compound prepared for tables.

Tests: synthetic laps `LapTime = 90.0 + 0.3*TyreLife` → `deg_slope ≈ 0.3` ±0.01; compound offset constant +2 s for MEDIUM vs SOFT fixture recovers ≈2; pit laps excluded.

### Task 5–7: components (no unit tests; visual smoke via app)

- `components/telemetry_plots.py`: `telemetry_figure(aligned: dict) -> go.Figure` — `make_subplots(rows=5, shared_xaxes=True)`: Speed, Delta, Throttle, Brake, Gear; driver colors passed through `aligned["colors"]`; `hovermode="x unified"`; dark layout.
- `components/track_map.py`: `track_map_figure(aligned, corners_df, microsectors) -> go.Figure` — equal-aspect outline, corner markers T<Number> at X/Y (cosmetic rotation applied), microsector segments colored by faster driver.
- `components/tables.py`: `corner_table(breakdown) -> go.Figure`, `stint_table(stats, offsets) -> go.Figure` — `go.Table`.

### Task 8: app.py + UI design pass

- `st.set_page_config` first; sidebar: year selectbox (2020..current), event selectbox (schedule-filtered), session code selectbox, driver A/B (candidates sorted by fastest lap), optional lap pickers populated from selected driver's laps.
- Caches: `load_events` (data), `load_session` (resource, keyed year/event/code), `align` (data, keyed incl. lap ids), `corners` (data), `tire stats` (data).
- Error states per Review Focus: sparse session, missing driver, telemetry gaps; `st.spinner` during first load.
- Dark pit-wall CSS + design direction via hallmark; then antislop-human contrast pass; web-design-guidelines final pass.

### Task 9: Verification

- End-to-end: fresh `streamlit run app.py`, cached 2023 Bahrain `R` and `Q`; each feature exercised (R-35 evidence list).
- Full pytest green; verification-before-completion report; skills review gate.
