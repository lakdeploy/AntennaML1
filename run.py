import os
import streamlit as st
import pandas as pd
import joblib
from modules.export_module import export_to_hfss_xml
from modules.gan_inverse_design import generate_design
from modules.viz_3d import show_3d_model
from streamlit_lottie import st_lottie
import json

# Load Lottie animation safely
def load_lottie(path="assets/animation.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

st.set_page_config(page_title="6G Antenna ML Project", layout="wide")

with st.sidebar:
    st_lottie(load_lottie(), height=150)
    tab = st.radio("Navigation", ["Home", "Predict", "Inverse Design", "Visualize"])

# Load pre-trained model
model = joblib.load("model/rf_model.pkl")

# Home tab
if tab == "Home":
    st.title("📡 6G AI-Powered Antenna Design")
    st.markdown("""
    This application allows:
    - 📈 Predicting Spectral Efficiency
    - 🎯 Inverse Design using GAN
    - 🧠 HFSS-compatible XML Export
    - 🧩 Real-time 3D Visualization
    """)

# Prediction tab
elif tab == "Predict":
    st.header("📈 Spectral Efficiency Prediction")

    col1, col2 = st.columns(2)
    with col1:
        length = st.number_input("Patch Length (mm)", 2.0, 20.0, 10.0)
        width = st.number_input("Patch Width (mm)", 2.0, 20.0, 8.0)
        substrate_height = st.number_input("Substrate Height (mm)", 0.1, 5.0, 1.6)
    with col2:
        dielectric = st.selectbox("Dielectric Constant", [2.2, 3.0, 4.4, 6.15, 10.2])
        feed_pos = st.slider("Feed Position (mm)", 0.0, 10.0, 2.0)

    input_df = pd.DataFrame([[length, width, substrate_height, dielectric, feed_pos]],
                            columns=["length", "width", "substrate_height", "dielectric_constant", "feed_position"])

    if st.button("🔍 Predict"):
        predicted = model.predict(input_df)[0]
        st.success(f"Predicted Spectral Efficiency: **{predicted:.4f} bits/s/Hz**")
        st.image("assets/prediction_plot.png", use_column_width=True)

# Inverse design tab
elif tab == "Inverse Design":
    st.header("🎯 Inverse Antenna Design")

    freq_target = st.slider("Target Frequency (GHz)", 1.0, 40.0, 28.0)
    if st.button("Generate Design"):
        design = generate_design(freq_target)
        st.json(design)

        if st.button("📤 Export to HFSS"):
            xml_file = export_to_hfss_xml(design)
            st.success(f"✅ HFSS XML Exported: `{xml_file}`")

# 3D visualization tab
elif tab == "Visualize":
    st.header("🧩 3D Antenna Model")
    try:
        show_3d_model("assets/antenna.obj", height=400)
    except Exception as e:
        st.error(f"❌ 3D rendering failed: {str(e)}")
