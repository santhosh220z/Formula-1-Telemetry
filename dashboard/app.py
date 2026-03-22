import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import load_model, simulate_stint, calculate_pit_window

st.set_page_config(page_title="F1 Strategy Predictor", layout="wide", page_icon="🏎️")

# Load external CSS
css_path = os.path.join(os.path.dirname(__file__), 'style.css')
with open(css_path, encoding="utf-8") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load HTML Header
html_path = os.path.join(os.path.dirname(__file__), 'header.html')
with open(html_path, encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

TRACKS = {
    "Bahrain International Circuit (Bahrain GP)": "Bahrain",
    "Jeddah Corniche Circuit (Saudi Arabian GP)": "Saudi Arabia",
    "Albert Park Circuit (Australian GP)": "Australia",
    "Suzuka Circuit (Japanese GP)": "Japan",
    "Shanghai International Circuit (Chinese GP)": "China",
    "Miami International Autodrome (Miami GP)": "Miami",
    "Imola Circuit (Emilia Romagna GP)": "Emilia Romagna",
    "Circuit de Monaco (Monaco GP)": "Monaco",
    "Circuit de Barcelona-Catalunya (Spanish GP)": "Spain",
    "Circuit Gilles Villeneuve (Canadian GP)": "Canada",
    "Red Bull Ring (Austrian GP)": "Austria",
    "Silverstone Circuit (British GP)": "Great Britain",
    "Hungaroring (Hungarian GP)": "Hungary",
    "Circuit de Spa-Francorchamps (Belgian GP)": "Belgium",
    "Circuit Zandvoort (Dutch GP)": "Netherlands",
    "Autodromo Nazionale Monza (Italian GP)": "Italy",
    "Marina Bay Street Circuit (Singapore GP)": "Singapore",
    "Circuit of the Americas (USA GP)": "United States",
    "Autódromo Hermanos Rodríguez (Mexico City GP)": "Mexico",
    "Autódromo José Carlos Pace (São Paulo GP)": "Brazil",
    "Las Vegas Strip Circuit (Las Vegas GP)": "Las Vegas",
    "Lusail International Circuit (Qatar GP)": "Qatar",
    "Yas Marina Circuit (Abu Dhabi GP)": "Abu Dhabi"
}

# Top Control Panel
with st.container():
    st.markdown("### Race Configuration")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        year = st.selectbox("Season Year", [2025, 2024, 2023, 2022, 2021], index=2) # Default 2023 since that's what was trained
    with col2:
        selected_track = st.selectbox("Grand Prix Event", list(TRACKS.keys()), index=0)
        event = TRACKS[selected_track]
    with col3:
        st.write("") # Spacer
        st.write("") # Spacer
        run_prediction = st.button("🚀 PREDICT")

st.markdown("---")

# Check for model existence
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', f'lstm_{year}_{event}.pth')

if run_prediction:
    if not os.path.exists(model_path):
        st.warning(f"⚠️ **Model not found:** No trained model available for {year} {event}.")
        st.info(f"Please run `python train.py --year {year} --event \"{event}\"` to train the model first.")
    else:
        with st.spinner("⏳ Analyzing telemetry and predicting tire wear..."):
            model, device = load_model(model_path)
            
            if model:
                # Mock a sequence for visualization
                mock_input_sequence = np.random.randn(5, 10)
                predicted_curve = simulate_stint(model, device, mock_input_sequence, future_laps=25)
                pit_lap = calculate_pit_window(predicted_curve, threshold=1.5)
                
                # Layout
                m_col1, m_col2, m_col3 = st.columns(3)
                
                base_time = predicted_curve[0]
                max_time = max(predicted_curve)
                degradation_delta = max_time - base_time
                status = "CRITICAL 🔴" if degradation_delta > 2.0 else "WARNING 🟡" if degradation_delta > 1.0 else "OPTIMAL 🟢"
                
                with m_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style='color:#888; margin:0;'>Initial Pace</h4>
                        <h2 style='color:white; margin:0;'>{base_time:.2f} <span style='font-size:1rem'>s</span></h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style='color:#888; margin:0;'>Expected Drop-off</h4>
                        <h2 style='color:#FF1801; margin:0;'>+{degradation_delta:.2f} <span style='font-size:1rem'>s / 25 laps</span></h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style='color:#888; margin:0;'>Tire Status</h4>
                        <h2 style='color:white; margin:0;'>{status}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visualization
                laps_x = list(range(1, len(predicted_curve) + 1))
                
                fig = go.Figure()
                
                # Main curve
                fig.add_trace(go.Scatter(
                    x=laps_x, y=predicted_curve,
                    mode='lines+markers',
                    name='Pace Drop-off',
                    line=dict(color='#FF1801', width=4),
                    marker=dict(size=8, color='white', line=dict(width=2, color='#FF1801')),
                    hovertemplate='Lap %{x}<br>Predicted Time: %{y:.2f}s<extra></extra>'
                ))
                
                if pit_lap:
                    # Pit Window Area
                    fig.add_vrect(
                        x0=pit_lap-1, x1=max(pit_lap+2, len(laps_x)),
                        fillcolor="rgba(255, 215, 0, 0.1)",
                        layer="below", line_width=0,
                        annotation_text="PIT WINDOW OPEN", annotation_position="top left",
                        annotation_font_color="gold"
                    )
                    
                    fig.add_vline(x=pit_lap, line_dash="dash", line_color="gold", line_width=2)
                    st.success(f"🏁 **Strategy Recommendation:** Box for fresh tires starting around **Lap {pit_lap}** to minimize race time lost to degradation.")
                
                fig.update_layout(
                    title=dict(text=f"Tire Degradation Forecast - {event} {year}", font=dict(size=20, color='white')),
                    xaxis=dict(title="Laps into Stint", showgrid=True, gridcolor='#333', color='white'),
                    yaxis=dict(title="Predicted Lap Time (s)", showgrid=True, gridcolor='#333', color='white'),
                    plot_bgcolor='#0E1117',
                    paper_bgcolor='#0E1117',
                    margin=dict(l=40, r=40, t=60, b=40),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Failed to load the model.")
else:
    # Empty state
    st.info("👆 Select a season year and track, then click **PREDICT** to generate the tire degradation telemetry.")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #666; font-size: 0.9em;'>Data backend powered by FastF1. Deep Learning architecture in PyTorch.</div>", unsafe_allow_html=True)
