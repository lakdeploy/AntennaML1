import pyvista as pv
import streamlit as st
import os

def show_3d_model(obj_path: str, height=400):
    try:
        if not os.path.exists(obj_path):
            st.error(f"File not found: {obj_path}")
            return

        # Load the mesh
        mesh = pv.read(obj_path)  # Automatically detects .obj, .stl, etc.

        # Create plotter
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh, color="lightblue", show_edges=True)

        # Export to HTML and display in iframe
        html_path = "temp_3d.html"
        plotter.export_html(html_path, backend='pythreejs')
        with open(html_path, "r") as f:
            html_content = f.read()

        st.components.v1.html(html_content, height=height)

    except Exception as e:
        st.error(f"❌ 3D rendering failed: {e}")
