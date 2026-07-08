import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import json
import random
import graphviz

# Ensure local src directory is visible to Python
sys.path.append(os.path.dirname(__file__))
from src.model import ThermalFNO

st.set_page_config(page_title="SciML Multi-Physics FNO", layout="wide")

# ---------------------------------------------------------
# 1. Load Normalization Statistics
# ---------------------------------------------------------
@st.cache_data
def load_stats():
    stats_path = "models/normalization_stats.json"
    if not os.path.exists(stats_path):
        st.error(f"Missing {stats_path}! Run train_fno.py first.")
        return None
    with open(stats_path, "r") as f:
        return json.load(f)

STATS = load_stats()

# ---------------------------------------------------------
# 2. Load Model (Strictly CPU for Streamlit Deployment)
# ---------------------------------------------------------
@st.cache_resource
def load_surrogate_model():
    device = torch.device("cpu")
    try:
        model = ThermalFNO(modes1=12, modes2=32, width=32).to(device)
        model.load_state_dict(torch.load("models/fno_best.pt", map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model. Ensure 'models/fno_best.pt' exists. Error: {e}")
        return None, None

model, device = load_surrogate_model()

# ---------------------------------------------------------
# 3. Dynamic Interactive Inference
# ---------------------------------------------------------
def run_interactive_inference(vel, length, offset):
    if STATS is None: return None, None, None
    
    nx, ny = 2212, 82
    x_min, x_max = 0.0, 0.22125
    y_min, y_max = -0.004075, 0.004075
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x_1d = np.linspace(x_min + dx/2, x_max - dx/2, nx)
    y_1d = np.linspace(y_min + dy/2, y_max - dy/2, ny)
    grid_x, grid_y = np.meshgrid(x_1d, y_1d)
    
    geom_mask = np.ones((ny, nx))
    fin_start_x = 0.00625
    fin_end_x = fin_start_x + length
    
    fin2_half_thick = 0.000625  
    f2_mask = (grid_x >= fin_start_x) & (grid_x <= fin_end_x) & \
              (grid_y >= offset - fin2_half_thick) & (grid_y <= offset + fin2_half_thick)
              
    fin1_y_bottom, fin1_y_top = y_min, y_min + 0.00125
    f1_mask = (grid_x >= fin_start_x) & (grid_x <= fin_end_x) & \
              (grid_y >= fin1_y_bottom) & (grid_y <= fin1_y_top)
              
    fin3_y_bottom, fin3_y_top = y_max - 0.00125, y_max
    f3_mask = (grid_x >= fin_start_x) & (grid_x <= fin_end_x) & \
              (grid_y >= fin3_y_bottom) & (grid_y <= fin3_y_top)
    
    geom_mask[f1_mask | f2_mask | f3_mask] = 0.0
                    
    u_inlet = np.full((ny, nx), vel)
    u_inlet_norm = (u_inlet - STATS['ui_min']) / (STATS['ui_max'] - STATS['ui_min'] + 1e-8)
    grid_x_norm = (grid_x - STATS['x_min']) / (STATS['x_max'] - STATS['x_min'] + 1e-8)
    grid_y_norm = (grid_y - STATS['y_min']) / (STATS['y_max'] - STATS['y_min'] + 1e-8)
    
    X_array = np.stack([geom_mask, u_inlet_norm, grid_x_norm, grid_y_norm], axis=0).astype(np.float32)
    X_tensor = torch.from_numpy(X_array).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(X_tensor).squeeze().cpu().numpy()
        
    pred_T = predictions[0] * (STATS['t_max'] - STATS['t_min']) + STATS['t_min']
    pred_U = predictions[1] * (STATS['uc_max'] - STATS['uc_min']) + STATS['uc_min']
    
    pred_U[geom_mask == 0.0] = 0.0 
    
    return pred_T, pred_U, geom_mask

# ---------------------------------------------------------
# 4. Main UI Layout
# ---------------------------------------------------------
st.title("SciML Multi-Physics FNO Thermal Surrogate Engine")
st.markdown("Real-time prediction of Fluid Dynamics and Heat Transfer, bypassing the OpenFOAM CFD solver.")

tab1, tab2, tab3 = st.tabs(["🎛️ Live Parameter Sweep", "📊 Architecture Overview", "🔍 CFD Ground Truth Validation"])

with tab1:
    st.subheader("Boundary & Geometric Parameters")
    slider_col1, slider_col2, slider_col3 = st.columns(3)
    
    with slider_col1:
        vel = st.slider("Inlet Velocity (m/s)", 0.10, 0.60, 0.35, step=0.01)
    with slider_col2:
        length = st.slider("Fin Length (m)", 0.010, 0.115, 0.050, step=0.005)
    with slider_col3:
        offset = st.slider("Middle Fin Y-Offset (m)", -0.0015, 0.0015, 0.0, step=0.0001, format="%.4f")

    st.markdown("---")
    
    if model is not None and STATS is not None:
        pred_T, pred_U, mask = run_interactive_inference(vel, length, offset)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'hspace': 0.3})
        physical_extent = [0.0, 0.22125, -0.004075, 0.004075]

        im_T = axes[0].imshow(
            pred_T, cmap='inferno', aspect='auto', origin='lower',
            extent=physical_extent, vmin=STATS['t_min'], vmax=STATS['t_max']
        )
        axes[0].set_title(f"Thermal Field Prediction (K) | Peak Temp: {np.max(pred_T):.1f} K", fontsize=12)
        axes[0].set_ylabel("Y (m)")
        fig.colorbar(im_T, ax=axes[0], fraction=0.02, pad=0.02, label="Temp (K)")
        
        im_U = axes[1].imshow(
            pred_U, cmap='viridis', aspect='auto', origin='lower',
            extent=physical_extent, vmin=STATS['uc_min'], vmax=STATS['uc_max']
        )
        axes[1].set_title(f"Velocity Field Prediction (m/s) | Peak |Velocity|: {np.max(np.abs(pred_U)):.2f} m/s", fontsize=12)
        axes[1].set_xlabel("X (m)")
        axes[1].set_ylabel("Y (m)")
        fig.colorbar(im_U, ax=axes[1], fraction=0.02, pad=0.02, label="Velocity (m/s)")
        
        st.pyplot(fig, use_container_width=True)

with tab2:
    st.subheader("Fourier Neural Operator (FNO) Architecture")
    st.markdown("""
    Unlike traditional CNNs that learn localized pixel kernels, the Fourier Neural Operator learns the **Green's function** of the governing Partial Differential Equations (PDEs) in the frequency domain. This allows it to evaluate fluid dynamics and heat transfer at any mesh resolution without retraining.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Macroscopic FNO Pipeline")
        st.markdown("The spatial inputs are lifted to a higher-dimensional latent space, passed through 4 frequency-domain convolutions, and projected back to physical physics scales.")
        
        # Macro Architecture Diagram
        macro_graph = graphviz.Digraph(node_attr={'shape': 'box', 'style': 'rounded,filled', 'fontname': 'Helvetica'})
        macro_graph.attr(rankdir='TB') # Top to Bottom
        
        macro_graph.node('in', 'Input Tensor X\n[4, 82, 2212]\n(Mask, Inlet_U, Grid_X, Grid_Y)', fillcolor='#E1F5FE')
        macro_graph.node('p', 'Linear Lifting Layer (P)\nChannels: 4 → 32', fillcolor='#FFF9C4')
        macro_graph.node('f1', 'Fourier Layer 1 (Modes: 12, 32)', fillcolor='#C8E6C9')
        macro_graph.node('f2', 'Fourier Layer 2 (Modes: 12, 32)', fillcolor='#C8E6C9')
        macro_graph.node('f3', 'Fourier Layer 3 (Modes: 12, 32)', fillcolor='#C8E6C9')
        macro_graph.node('f4', 'Fourier Layer 4 (Modes: 12, 32)', fillcolor='#C8E6C9')
        macro_graph.node('q', 'Projection Layer (Q)\nMLP: 32 → 128 → 2', fillcolor='#FFF9C4')
        macro_graph.node('out', 'Output Tensor Y\n[2, 82, 2212]\n(Temperature, Velocity)', fillcolor='#FCE4EC')
        
        macro_graph.edge('in', 'p')
        macro_graph.edge('p', 'f1')
        macro_graph.edge('f1', 'f2')
        macro_graph.edge('f2', 'f3')
        macro_graph.edge('f3', 'f4')
        macro_graph.edge('f4', 'q')
        macro_graph.edge('q', 'out')
        
        st.graphviz_chart(macro_graph, use_container_width=True)

    with col2:
        st.markdown("### 2. Microscopic Spectral Convolution")
        st.markdown("Inside each Fourier block, the signal splits. The upper branch performs operations in the Fourier domain to capture global physics, while the lower branch acts as a spatial bypass.")
        
        # Micro Architecture Diagram
        micro_graph = graphviz.Digraph(node_attr={'shape': 'box', 'style': 'rounded,filled', 'fontname': 'Helvetica'})
        micro_graph.attr(rankdir='TB')
        
        micro_graph.node('v_in', 'Input v(x)', fillcolor='#EEEEEE')
        
        # Global Branch (Frequency Domain)
        micro_graph.node('fft', '2D Fast Fourier Transform\n(FFT)', fillcolor='#D1C4E9')
        micro_graph.node('trunc', 'Truncate High Frequencies\n(Keep lowest 12x32 modes)', fillcolor='#D1C4E9')
        micro_graph.node('mul', 'Complex Weight Multiplication\n(Learnable PDE Operator)', fillcolor='#D1C4E9')
        micro_graph.node('ifft', '2D Inverse FFT\n(iFFT)', fillcolor='#D1C4E9')
        
        # Local Branch (Spatial Domain)
        micro_graph.node('w', 'Local Linear Transform (W)\n(1x1 Spatial Convolution)', fillcolor='#FFE0B2')
        
        # Addition and Activation
        micro_graph.node('add', 'Element-wise Addition (+)', shape='circle', fillcolor='#CFD8DC')
        micro_graph.node('act', 'GELU Activation', fillcolor='#FFF9C4')
        micro_graph.node('v_out', 'Output v(x\')', fillcolor='#EEEEEE')
        
        # Edges Global
        micro_graph.edge('v_in', 'fft')
        micro_graph.edge('fft', 'trunc')
        micro_graph.edge('trunc', 'mul')
        micro_graph.edge('mul', 'ifft')
        micro_graph.edge('ifft', 'add')
        
        # Edges Local
        micro_graph.edge('v_in', 'w')
        micro_graph.edge('w', 'add')
        
        # Edges Out
        micro_graph.edge('add', 'act')
        micro_graph.edge('act', 'v_out')
        
        st.graphviz_chart(micro_graph, use_container_width=True)

    st.markdown("---")
    st.subheader("Physical Modeling Constraints")
    st.markdown("""
    * **Boundary Layer Adherence:** Masking enforces absolute no-slip conditions (`Velocity = 0.0 m/s`) within the solid fin boundaries.
    * **Conjugate Heat Transfer (CHT):** Solid fin domains are simulated with a constant volumetric heat generation rate. This rate is fixed globally across the simulation dataset and acts as the implicit driving thermal force for the PDEs.
    * **Spatial Mapping:** By utilizing actual coordinates `(X, Y)` as input channels alongside the geometry mask, the FNO successfully maps fluid dynamics without requiring traditional body-fitted meshing configurations.
    """)

with tab3:
    st.subheader("Holdout Set: FNO vs. OpenFOAM (Unseen Data)")
    st.markdown("""
    To prove generalization across the operational envelope, the FNO was evaluated against completely unseen parameter combinations. 
    The AI surrogate independently predicts both Conjugate Heat Transfer (CHT) and Fluid Velocity profiles with high fidelity.
    
    *Note: Absolute Error is tracked independently for fluid domains and solid domains to ensure rigorous modeling of internal fin conduction.*
    """)
    
    if st.button("🔄 Load Random Test Case", type="primary"):
        available_images = [f"assets/val_case_{i}.png" for i in range(1, 4)]
        existing_images = [img for img in available_images if os.path.exists(img)]
        
        if existing_images:
            chosen_image = random.choice(existing_images)
            st.image(chosen_image, use_container_width=True)
            st.success(f"Successfully loaded validation sample: `{os.path.basename(chosen_image)}`")
        else:
            st.error("No validation images found in the `assets/` directory. Run `scripts/generate_val_plots.py` locally and push the resulting PNGs to your repository.")