import streamlit as st
import pandas as pd
import plotly.express as px
from prediction_module import load_model

def show_graph():
    st.title("📈 Feature Correlation")
    df = load_model()["df"]
    param = st.selectbox("Choose parameter", df.columns.drop("spectral_efficiency"))
    fig = px.scatter(df, x=param, y="spectral_efficiency",
                     trendline="ols", title=f"{param} vs Spectral Efficiency")
    st.plotly_chart(fig, use_container_width=True)
