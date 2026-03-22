import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import load_model, simulate_stint, calculate_pit_window

st.set_page_config(page_title="F1 Telemetry Predictor", layout="wide", page_icon="🏎️")

st.title("🏎️ Formula 1 Telemetry & Tire Degradation Predictor")
st.markdown("Predict future lap performance, tire degradation, and optimal pit stop windows based on historical F1 telemetry data.")

# Set up sidebar
st.sidebar.header("Configuration")
year = st.sidebar.selectbox("Year", [2023, 2022, 2021], index=0)
event = st.sidebar.selectbox("Event", ["Bahrain", "Monaco", "Silverstone", "Monza", "Abu Dhabi"], index=0)

# Check for model existence
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', f'lstm_{year}_{event}.pth')

if st.sidebar.button("Load Model & Run Prediction"):
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train the model first.")
    else:
        with st.spinner("Loading Model & Running Predictions..."):
            model, device = load_model(model_path)
            
            if model:
                st.success("Model loaded successfully!")
                
                # Mock a sequence for visualization (since we don't have live telemetry hooked up here)
                # Shape: (5, 10) - 5 laps, 10 features
                mock_input_sequence = np.random.randn(5, 10)
                
                # Generate predictions
                predicted_curve = simulate_stint(model, device, mock_input_sequence, future_laps=20)
                
                # Calculate pit window
                pit_lap = calculate_pit_window(predicted_curve, threshold=1.5)
                
                # 1. Visualization: Tire Degradation Curve
                st.subheader("Tire Degradation Forecast")
                laps_x = list(range(1, len(predicted_curve) + 1))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=laps_x, 
                    y=predicted_curve,
                    mode='lines+markers',
                    name='Predicted Lap Time',
                    line=dict(color='red', width=3)
                ))
                
                # Add pit window indicator
                if pit_lap:
                    fig.add_vline(x=pit_lap, line_dash="dash", line_color="yellow", 
                                  annotation_text="Optimal Pit Window", annotation_position="top right")
                    
                    st.info(f"💡 Recommendation: Box box! Optimal pit stop window opens around **Lap {pit_lap}** into the stint.")
                else:
                    st.success("Tires are holding up well. No immediate pit stop recommended.")

                fig.update_layout(
                    title="Predicted Lap Time Drop-off",
                    xaxis_title="Laps in Stint",
                    yaxis_title="Predicted Lap Time (Seconds)",
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. Metric Cards
                col1, col2, col3 = st.columns(3)
                base_time = predicted_curve[0]
                max_time = max(predicted_curve)
                degradation_delta = max_time - base_time
                
                col1.metric("Initial Stint Pace", f"{base_time:.2f} s")
                col2.metric("Max Degradation (20 laps)", f"+{degradation_delta:.2f} s", delta_color="inverse")
                
                status = "Critical" if degradation_delta > 2.0 else "Stable"
                col3.metric("Tire Status", status)
                
            else:
                st.error("Failed to load the model.")
                
st.markdown("---")
st.markdown("Data powered by [FastF1](https://github.com/theOehrly/Fast-F1). Model structure: LSTM in PyTorch.")
