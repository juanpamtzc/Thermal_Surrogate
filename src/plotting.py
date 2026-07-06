import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

def plot_fields_2x1(foam_file: str, time_step: float = None, save_path: str = None):
    """Plots Temperature and Velocity fields using PyVista."""
    reader = pv.OpenFOAMReader(foam_file)
    reader.enable_all_cell_arrays()
    reader.set_active_time_value(time_step if time_step else reader.time_values[-1])
    mesh = reader.read()
    
    temp_blocks = pv.MultiBlock([b for m in mesh for b in (m if isinstance(m, pv.MultiBlock) else [m]) if isinstance(b, pv.UnstructuredGrid) and "T" in b.array_names])
    vel_blocks = pv.MultiBlock([b for m in mesh for b in (m if isinstance(m, pv.MultiBlock) else [m]) if isinstance(b, pv.UnstructuredGrid) and "U" in b.array_names])

    pv.set_jupyter_backend('static')
    plotter = pv.Plotter(shape=(2, 1), window_size=[1000, 800])
    
    plotter.subplot(0, 0)
    plotter.add_text("Temperature Field (T)", font_size=10)
    if temp_blocks: plotter.add_mesh(temp_blocks, scalars="T", cmap="turbo", show_edges=False)
    plotter.view_xy(); plotter.camera.zoom(1.2)
    
    plotter.subplot(1, 0)
    plotter.add_text("Velocity Magnitude (U)", font_size=10)
    if vel_blocks: plotter.add_mesh(vel_blocks, scalars="U", cmap="viridis", show_edges=False)
    plotter.view_xy(); plotter.camera.zoom(1.2)
    
    if save_path: plotter.show(screenshot=save_path)
    else: plotter.show()

def plot_local_distribution(data_dict: Dict[str, Dict[str, np.ndarray]], target_key: str, ylabel: str, title: str):
    """Generic function to plot local variables (T or q_mag) along the X-axis."""
    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for label, arrays in data_dict.items():
        x, y = arrays['X'], arrays[target_key]
        sort_idx = np.argsort(x)
        x_sorted, y_sorted = x[sort_idx], y[sort_idx]
        
        avg_y = np.trapz(y_sorted, x_sorted) / (x_sorted[-1] - x_sorted[0])
        p = plt.plot(x_sorted, y_sorted, linewidth=2, label=f"{label} (Local)")
        plt.axhline(avg_y, color=p[0].get_color(), linestyle='--', alpha=0.7, label=f"{label} (Avg: {avg_y:.2f})")

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_teertstra_comparison(Re_cfd: float, Nu_cfd: float, paper_data: np.ndarray, paper_data_2: np.ndarray):
    """Plots the CFD output against digitized Teertstra (2000) data."""
    plt.figure(figsize=(9, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(paper_data[:, 0], paper_data[:, 1], 'k--', linewidth=2, label="Teertstra et al. (Model 1)")
    plt.plot(paper_data_2[:, 0], paper_data_2[:, 1], 'b--', linewidth=2, label="Teertstra et al. (Model 2)")
    plt.plot(Re_cfd, Nu_cfd, marker='*', color='red', markersize=15, linestyle='None', label="OpenFOAM CFD")
    
    plt.title("Nusselt Number vs. Modified Channel Reynolds Number", fontsize=14, fontweight='bold')
    plt.xlabel("Modified Channel Reynolds Number ($Re^*$)", fontsize=12)
    plt.ylabel("Channel Nusselt Number ($Nu_b$)", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.show()