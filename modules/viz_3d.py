import os
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import streamlit as st


def show_3d_model(obj_path, height=400):
    try:
        if not os.path.exists(obj_path):
            st.warning(f"⚠️ File not found: {obj_path}")
            return

        mesh = trimesh.load(obj_path)

        # Convert mesh to 3D faces
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        mesh_faces = mesh.vertices[mesh.faces]
        ax.add_collection3d(Poly3DCollection(mesh_faces, facecolors='lightblue', edgecolor='gray', linewidths=0.2, alpha=0.9))

        # Autoscale
        scale = mesh.vertices.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        ax.set_axis_off()

        # Save and show
        screenshot_path = "assets/3d_render.png"
        plt.tight_layout()
        plt.savefig(screenshot_path)
        plt.close()

        st.image(screenshot_path, caption="📡 3D Antenna View", use_column_width=True)

    except Exception as e:
        st.error(f"❌ 3D rendering failed: {e}")
