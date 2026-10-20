# AGENTS.md — Pit Wall Telemetry

## Project state (read first)

- **Greenfield rebuild from a written spec.** The working tree starts intentionally empty. The previous implementation was deleted and is recoverable only via `git show 4a944d9` (old stack: PyTorch LSTM tire predictor, `train*.py`, `utils/`, `dashboard/`).
- Do **NOT** restore old files — the ML-training design is superseded. This rebuild is a pure telemetry analytics dashboard with no training pipeline. Old code is reference-only for FastF1 usage patterns; where it conflicts with the spec below, the spec wins.
- Since the tree is empty, this file's "Product contract" section is the specification of record until code exists.

## Product contract

"Pit Wall Telemetry": F1 dashboard where the user selects season (2020–present), GP, session (Q, R, FP1–FP3) and two drivers, then gets:

1. **Synced 5-trace telemetry comparison** — Speed, Delta Time, Throttle, Brake, Gear. Plotly `make_subplots`, shared x-axis (distance), linked zoom/pan, unified crosshair.
2. **Track map** — 2D circuit outline from GPS data with corner apex markers (T1, T2 …) and a mini-sector advantage heat map (each segment colored by the faster driver).
3. **Corner-by-corner analysis** — per corner: minimum corner speed (V_min), braking onset point, throttle pick-up distance, corner delta between the two drivers.
4. **Stint-level tire degradation** — lap-time evolution modeling per stint, with compound normalization.

## Stack

- Python 3.10+ · Streamlit · FastF1 · NumPy/Pandas/SciPy (SciPy for `interp1d` distance-grid resampling and numerical gradients) · Plotly **graph objects only** for all charts (no `st.line_chart`, no matplotlib).
- Persistence: FastF1 disk cache in `.fastf1_cache/` + Streamlit caching: `st.cache_resource` for loaded fastf1 Session objects (they are not picklable — `st.cache_data` will fail on them), `st.cache_data` for derived DataFrames.

## Target structure

```
core/                     # pure Python: fastf1 + numpy/pandas/scipy; NO streamlit imports
  session_manager.py      # session loading, caching, metadata
  telemetry_resampler.py  # distance-based resampling; aligns both drivers on a common grid; delta-time math
  corner_analyzer.py      # micro-sector + apex metrics
  tire_model.py           # stint degradation & compound normalization
components/               # Plotly figure builders: take core outputs, return go.Figure
  telemetry_plots.py      # synced 5-row chart
  track_map.py            # circuit outline + advantage map
  tables.py               # corner breakdown & stint stats
app.py                    # Streamlit wiring only (widgets, layout, st.* calls, CSS)
```

- Data flow: `app.py` → `core.session_manager` → `core.telemetry_resampler` (both drivers) → `core.corner_analyzer` / `core.tire_model` → `components.*` → figures rendered in `app.py`.
- Imports are plain top-level (`from core.telemetry_resampler import ...`) — `streamlit run app.py` runs from repo root, so no `sys.path` hacks (the old dashboard needed them; this layout does not).
- Keep all math/analysis in `core/`, all plotting in `components/`; never call `fastf1` from `app.py` or `components/` directly.

## Commands (dev box: Windows / PowerShell)

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py
```

- No test/lint/typecheck config yet. While building, smoke-test modules with a `__main__` guard (`python core/corner_analyzer.py`) or `python -c "from core.x import f; ..."`. Add real tests before this section gets stale.
- If FastF1 acts oddly after code changes, clear the cache (`Remove-Item -Recurse -Force .fastf1_cache`) before assuming a code bug.
- Create `.gitignore` covering `.fastf1_cache/`, `venv/`, `__pycache__/` before the first commit — cached telemetry dumps will bloat the repo.

## FastF1 gotchas

- Every new (year, GP, session) triple downloads from the F1 API on first load: slow (tens of seconds) and requires internet. Subsequent loads come from disk cache.
- **F1 gates the livetiming archive behind an authenticated F1 TV account** (since ~15 Oct 2026, Fast-F1 issue #978). We pin `fastf1==3.9.0a0`, which added auth support. Users sign in once with `venv\Scripts\python -m fastf1 auth f1tv --authenticate` (needs an F1 TV Access/Pro/Premium subscription; token listed by `--status`, lives ~4 days). Without it, loads finish but `laps`/`get_car_data()` raise `DataNotLoadedError`; results-only data is still free (Jolpica/Ergast). `app.py` surfaces this with `_auth_hint` on data failures.
- Call `fastf1.Cache.enable_cache(...)` (pointing at `.fastf1_cache`) before any `session.load()`.
- **Never hardcode a GP-name map** (the old code shipped one and it was brittle/stale). Build dropdowns from `fastf1.get_event_schedule(year)` and resolve with `fastf1.get_session(year, event, code)`.
- Session codes: `'Q'`, `'R'`, `'FP1'`…`'FP3'`.
- Apex markers: `session.get_circuit_info().corners` — X/Y are in the telemetry coordinate frame, so the map outline and markers align directly; `circuit_info.rotation` is only for cosmetic map orientation.
- Pin dependencies with `==` in `requirements.txt` — fastf1's API changes between minor versions (e.g. plotting/color helpers changed in 3.4+).

## Streamlit gotchas

- `st.set_page_config(...)` must be the first `st` call in `app.py`.
- Every widget interaction reruns the whole script top-to-bottom: never fetch or compute telemetry outside `st.cache_*`. Key caches on (year, event, session, driver pair).
- Load the Session object once with `st.cache_resource`; everything downstream (resampled frames, corner tables) as `st.cache_data`.

## Conventions

- This file is the onboarding source of truth. Update it when structure, commands, or cache behavior change; keep it compact.

## Skills (installed agent skills — load via the `skill` tool)

Code (any task):
- `antislop` + `ponytail` — before writing or reviewing any code.
- `test-driven-development` — before implementing any `core/` module.
- `systematic-debugging` — before fixing any bug or odd telemetry result.
- `verification-before-completion` — before claiming a feature works/done.
- `writing-plans` — before a multi-step build sprint off the spec above.

UI/UX (`app.py`, `components/`):
- `hallmark` — lead the one-time greenfield design pass for the pit-wall look; `antislop-ui` is the ongoing filter afterward.
- `ui-ux-pro-max` — searchable reference (palettes, type, chart guidance) when styling.
- `antislop-human` — contrast/accessibility pass on the dark theme.
- `web-design-guidelines` — final UI review pass.
