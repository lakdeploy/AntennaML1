import streamlit as st
import joblib
import os
import pandas as pd
from modules.export_module import export_to_hfss_xml
from modules.gan_inverse_design import generate_design
from modules.viz_3d import show_3d_model
from streamlit_lottie import st_lottie
import json

st.set_page_config(layout="wide")

tabs = st.tabs(["Home", "Predict", "Inverse Design"])
home, predict_tab, inverse_tab = tabs

with home:
    st.title("📡 6G Antenna ML Suite")
    st.write("Multi-functional app: prediction, inverse design, 3D preview, export.")

    # Add Lottie animation
    from streamlit_lottie import st_lottie
    import json
    try:
        with open("assets/success_animation.json") as f:
            anim = json.load(f)
            st_lottie(anim, height=200, speed=0.1)
    except:
        st.warning("⚠️ success_animation.json not found in assets/")


with predict_tab:
    st.header("🎯 Forward Prediction")

    model = joblib.load("model/rf_model.pkl")
    labels = ["antenna_array_size", "transmission_power", "beamwidth", "channel_gain",
              "interference_level", "user_density", "carrier_frequency", "mobility_speed"]

    # Initialize session state
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
        st.session_state.prediction_valid = False
        st.session_state.prev_inputs = {}

    # User input sliders
    cols = st.columns(2)
    user_input = {}
    for i, label in enumerate(labels):
        val = cols[i % 2].slider(label.replace('_', ' ').title(), 0, 1000, 100, key=f"slider_{label}")
        user_input[label] = val

    # Detect change in input and invalidate prediction
    if user_input != st.session_state.prev_inputs:
        st.session_state.prediction_valid = False
    st.session_state.prev_inputs = user_input

    # Prediction Button
    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        st.session_state.prediction = pred
        st.session_state.prediction_valid = True

    # Display prediction even if invalidated
    if st.session_state.prediction is not None:
        color = (
            "green" if st.session_state.prediction > 12 else
            "orange" if st.session_state.prediction > 6 else
            "red"
        )

        # If inputs have changed and prediction not re-run
        if not st.session_state.prediction_valid:
            color = "red"

        st.markdown(
            f"<div style='padding: 1rem; border: 2px solid {color}; border-radius: 10px;'>"
            f"<h4 style='color: {color};'>Predicted Spectral Efficiency: {st.session_state.prediction:.2f} bps/Hz</h4>"
            f"<small>{'⚠️ Prediction may be outdated due to input change' if not st.session_state.prediction_valid else ''}</small>"
            "</div>", unsafe_allow_html=True
        )

with inverse_tab:
    st.header("🧠 Inverse Design via GAN")
    freq = st.number_input("Desired Frequency (GHz)", 28.0, 83.0, 60.0, step=1.0)
    num = st.slider("Number of Designs", 1, 5, 1)

    if st.button("Generate Designs"):
        designs = generate_design(freq, num)
        dfd = pd.DataFrame(designs, columns=["Length_mm", "Width_mm", "Height_mm"])
        st.table(dfd)

        # Show 3D design if OBJ exists
        # st.info("👀 OBJ viewer: Upload your custom 3D model to `assets/antenna.obj` to visualize it.")
        if os.path.exists("assets/antenna.obj"):
            show_3d_model("assets/antenna.obj", height=400)
        else:
            st.warning("⚠️ OBJ file not found! Please add it as `assets/antenna.obj`.")

