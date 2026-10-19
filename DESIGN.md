# DESIGN.md — Pit Wall Telemetry

Direction supplied by the owner ("dark pit-wall UI aesthetic"). Transcribed; not invented.

- **Identity:** pit-wall control room. Charts + data are the identity; chrome stays out of the way.
- **Tokens (single source `style.css :root`, mirrored in `components/telemetry_plots.py`):**
  - `--pit-bg #0B0D10` page · `--pit-panel #0E1116` surface · `--pit-line #23262B` hairlines/grid
  - `--pit-text #D7DADE` primary text · `--pit-muted #8B9097` secondary text
  - `--pit-accent #FF1801` F1 red — allowed on: delta trace, winner highlights, primary CTA only
- **Type:** Segoe UI (roman, system, offline-safe) for display + body; Consolas for numeric readouts (tabular). No italic headers.
- **Driver colors:** never decorative; always from FastF1 driver palette (falls back to a fixed neutral set).
- **Motion:** none beyond Streamlit transitions and unified hover crosshairs (MOTION 1).
- **States:** every data view ships empty / loading / error variants (R-27).
- **Forbidden:** gradients, glows, glassmorphism, grid background textures, fabricated stats, icons-as-decoration.
- **Accent discipline:** max 2-3 core colors + 1 accent (R-29); greys are neutral, not "core".
