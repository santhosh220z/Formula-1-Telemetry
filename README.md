# Pit Wall Telemetry

F1 telemetry analytics dashboard: select a season (2020–present), Grand Prix, and session (Q, R, FP1–FP3), then compare two drivers point-by-point.

## Run

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py
```

## F1 account authentication (required)

F1 has gated all timing and telemetry data behind an authenticated F1 TV account (effective mid-October 2026; results data only stayed free). Sign in once:

```powershell
venv\Scripts\python -m fastf1 auth f1tv --authenticate
```

Enter your F1 TV email and password when prompted. An active F1 TV Access, Pro or Premium subscription is required. The token lives about four days, then run the command again. Check status with `venv\Scripts\python -m fastf1 auth f1tv --status`.

## Features

1. Synced 5-trace telemetry comparison (Speed, Delta Time, Throttle, Brake, Gear)
2. Track map with corner apex markers and mini-sector driver advantage heat map
3. Corner-by-corner analysis (V_min, braking onset, throttle pick-up, corner delta)
4. Stint-level tire degradation modeling

Data via [FastF1](https://docs.fastf1.dev). Without an F1 TV sign-in the app loads the calendar and results, but telemetry panels show an error explaining how to authenticate (see above). First load of a session downloads from the F1 API (needs internet); after that it's served from `.fastf1_cache/`.
