import os
import json
import streamlit as st
import pandas as pd
import joblib
from modules.export_module import export_to_hfss_xml
from modules.gan_inverse_design import generate_design
from modules.viz_3d import show_3d_model
from streamlit_lottie import st_lottie

# ---------------------- Lottie Loader ----------------------
def load_lottie(path="assets/animation.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="6G Antenna ML Project", layout="wide")

with st.sidebar:
    try:
        animation_data = load_lottie()
        if animation_data:
            st_lottie(animation_data, height=150)
        else:
            st.info("No animation found.")
    except Exception as e:
        st.warning("⚠️ Animation failed to load.")

    tab = st.radio("Navigation", ["Home", "Predict", "Inverse Design", "Visualize"])

# ---------------------- Load Model ----------------------
try:
    model = joblib.load("model/rf_model.pkl")
except Exception as e:
    st.error("❌ Could not load the prediction model.")
    st.stop()

# ---------------------- Home Tab ----------------------
if tab == "Home":
    st.title("📡 6G AI-Powered Antenna Design")
    st.markdown("""
    Welcome to the AI-driven antenna design tool for next-generation communication. This app enables:

    - 📈 **Predicting Spectral Efficiency**
    - 🎯 **Inverse Antenna Design using GAN**
    - 🧠 **HFSS-Compatible XML Export**
    - 🧩 **3D Antenna Visualization (local only)**
    """)

# ---------------------- Predict Tab ----------------------
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

# ---------------------- Inverse Design Tab ----------------------
elif tab == "Inverse Design":
    st.header("🎯 Inverse Antenna Design with GAN")

    freq_target = st.slider("🎯 Target Frequency (GHz)", 1.0, 40.0, 28.0)
    if st.button("Generate Design"):
        try:
            design = generate_design(freq_target)
            st.json(design)

            if st.button("📤 Export to HFSS"):
                xml_file = export_to_hfss_xml(design)
                st.success(f"✅ HFSS XML Exported: `{xml_file}`")
        except Exception as e:
            st.error(f"❌ Design generation failed: {str(e)}")

# ---------------------- 3D Visualization Tab ----------------------
elif tab == "Visualize":
    st.header("🧩 3D Antenna Visualization")
    try:
        show_3d_model("assets/antenna.obj", height=400)
    except Exception as e:
        st.warning("⚠️ 3D rendering is not supported in Streamlit Cloud.")
        st.image("assets/3d_placeholder.png", caption="3D view not available")
