# Pit Wall Telemetry

F1 telemetry analytics dashboard: select a season (2020–present), Grand Prix, and session (Q, R, FP1–FP3), then compare two drivers point-by-point.

## Run

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py
```

## Features

1. Synced 5-trace telemetry comparison (Speed, Delta Time, Throttle, Brake, Gear)
2. Track map with corner apex markers and mini-sector driver advantage heat map
3. Corner-by-corner analysis (V_min, braking onset, throttle pick-up, corner delta)
4. Stint-level tire degradation modeling

Data via [FastF1](https://docs.fastf1.dev). First load of a session downloads from the F1 API (needs internet); after that it's served from `.fastf1_cache/`.
