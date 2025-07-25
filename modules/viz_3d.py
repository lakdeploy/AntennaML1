# modules/viz_3d.py

import pyvista as pv
import streamlit as st
import os

def show_3d_model(path, height=400):
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return

    try:
        # Force PyVista to off-screen mode
        pv.OFF_SCREEN = True

        mesh = pv.read(path)
        plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
        plotter.add_mesh(mesh, color='lightblue', show_edges=True)
        plotter.set_background('white')

        html_path = "assets/plot.html"
        plotter.export_html(html_path, backend='none', offline=True)

        # Read and embed HTML
        with open(html_path, "r") as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=height)

    except Exception as e:
        st.error(f"❌ 3D rendering failed: {e}")
