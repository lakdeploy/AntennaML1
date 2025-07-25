import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from streamlit_lottie import st_lottie

def predict_efficiency():
    st.title("🎯 Predict Spectral Efficiency")

    model_bundle = joblib.load("models/frequency_predictor.pkl")
    model = model_bundle["model"]

    col1, col2 = st.columns(2)
    with col1:
        antenna_size = st.slider("Antenna Array Size", 8, 128, 64, 8)
        tx_power = st.slider("Transmission Power (dBm)", 10, 40, 25)
        beamwidth = st.slider("Beamwidth (°)", 5, 60, 30)
        gain = st.slider("Channel Gain (dB)", -140, -60, -100)
    with col2:
        interference = st.slider("Interference Level", 0.0, 1.0, 0.5)
        users = st.slider("User Density (per km²)", 10, 1000, 500)
        freq = st.slider("Carrier Frequency (GHz)", 28, 83, 60)
        speed = st.slider("Mobility (km/h)", 0, 120, 60)

    input_data = {
        "antenna_array_size": antenna_size,
        "transmission_power": tx_power,
        "beamwidth": beamwidth,
        "channel_gain": gain,
        "interference_level": interference,
        "user_density": users,
        "carrier_frequency": freq,
        "mobility_speed": speed,
    }
    input_df = pd.DataFrame([input_data])

    if "last_input" not in st.session_state:
        st.session_state.last_input = None
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    # Compare input to last_input
    inputs_changed = st.session_state.last_input != input_data

    if st.button("🔮 Predict"):
        pred = model.predict(input_df)[0]
        st.session_state.last_prediction = pred
        st.session_state.last_input = input_data
        color = "green" if pred > 12 else "orange" if pred > 6 else "red"
        st.markdown(
            f"<div style='padding: 1rem; border: 2px solid {color}; border-radius: 10px;'>"
            f"<h4 style='color: {color};'>Predicted Spectral Efficiency: {pred:.2f} bps/Hz</h4>"
            "</div>", unsafe_allow_html=True
        )
        with open("assets/success_animation.json") as f:
            anim = json.load(f)
        st_lottie(anim, height=180)
    elif st.session_state.last_prediction is not None:
        # Show old prediction in RED if input changed
        old_color = "red" if inputs_changed else (
            "green" if st.session_state.last_prediction > 12 else
            "orange" if st.session_state.last_prediction > 6 else
            "red"
        )
        st.markdown(
            f"<div style='padding: 1rem; border: 2px dashed {old_color}; border-radius: 10px;'>"
            f"<h4 style='color: {old_color};'>Last Predicted Efficiency: {st.session_state.last_prediction:.2f} bps/Hz</h4>"
            f"<small>{'⚠️ Change detected. Re-run prediction.' if inputs_changed else ''}</small>"
            "</div>", unsafe_allow_html=True
        )
