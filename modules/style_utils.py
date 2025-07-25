import streamlit as st

def set_theme():
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
        }
        .stButton > button {
            background-color: #FF4B4B;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
