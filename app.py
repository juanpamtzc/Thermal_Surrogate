import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# Ensure local src directory is visible to Python
sys.path.append(os.path.dirname(__file__))
from src.model import ThermalFNO

st.set_page_config(page_title="SciML Thermal Surrogate", layout="wide")

# ---------------------------------------------------------
# 1. Old Model Statistics (UPDATE THESE LATER)
# ---------------------------------------------------------
# These are dummy stats to prevent crashes. The model will run, 
# but the predictions might look weird until you put your actual old stats here!
STATS = {
    't_min': 293.15, 't_max': 380.0, 
    'u_min': 0.1,    'u_max': 0.6,
    'x_min': 0.0,    'x_max': 0.2,
    'y_min': -0.05,  'y_max': 0.05,
}

# ---------------------------------------------------------
# 2. Model Loading
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
# 3. Dynamic Interactive Generation Helper
# ---------------------------------------------------------
def run_interactive_inference(vel, length, offset):
    """Generates an on-the-fly grid so no external files are needed."""
    # 1. Create a basic 64x64 analytical grid
    x = np.linspace(0, 0.2, 64)
    y = np.linspace(-0.05, 0.05, 64)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # 2. Dummy Geometry Mask (All 1s for now, replace with your fin logic later)
    geom_mask = np.ones((64, 64))
    
    # 3. Assign velocity channel
    u_inlet = np.zeros((64, 64))
    u_inlet[:, 0] = vel  # Assuming inlet is at x=0
    
    # 4. Normalize
    u_inlet_norm = (u_inlet - STATS['u_min']) / (STATS['u_max'] - STATS['u_min'] + 1e-8)
    grid_x_norm = (grid_x - STATS['x_min']) / (STATS['x_max'] - STATS['x_min'] + 1e-8)
    grid_y_norm = (grid_y - STATS['y_min']) / (STATS['y_max'] - STATS['y_min'] + 1e-8)
    
    # 5. Stack and run forward pass
    input_channels = np.stack([geom_mask, u_inlet_norm, grid_x_norm, grid_y_norm], axis=0)
    X_tensor = torch.from_numpy(input_channels).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_T_norm = model(X_tensor).squeeze().cpu().numpy()
        
    # 6. Reverse normalization back to Kelvin
    pred_T = pred_T_norm * (STATS['t_max'] - STATS['t_min']) + STATS['t_min']
    return pred_T

# ---------------------------------------------------------
# 4. Main UI Layout
# ---------------------------------------------------------
st.title("⚙️ SciML Fourier Neural Operator (FNO) Surrogate Engine")
st.markdown("Replacing heavy OpenFOAM CFD solvers with millisecond AI inference.")

tab1, tab2 = st.tabs(["🎛️ Live Parameter Sweep", "📊 Holdout Validation Scenarios"])

with tab1:
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.subheader("Boundary Parameters")
        vel = st.slider("Inlet Velocity (m/s)", 0.1, 0.6, 0.35, step=0.01)
        length = st.slider("Fin Length (m)", 0.010, 0.115, 0.050, step=0.005)
        offset = st.slider("Fin Y-Center Offset (m)", -0.0015, 0.0015, 0.0, step=0.0005)

    with col2:
        st.subheader("FNO Predicted Temperature Field")
        if model is not None:
            pred_T = run_interactive_inference(vel, length, offset)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(pred_T, cmap='inferno', aspect='auto', origin='lower')
            fig.colorbar(im, ax=ax, label="Temperature (K)")
            ax.set_title(f"Instantaneous Thermal Mapping (Inlet Vel: {vel:.2f} m/s)")
            st.pyplot(fig)
        else:
            st.warning("Model weights not loaded. Check the error message above.")

with tab2:
    st.subheader("Model Validation against Ground Truth CFD")
    
    # --- PLACEHOLDER FOR PNG LOGIC ---
    st.info("🚧 **Simulations Running** 🚧 \n\nA new OpenFOAM dataset is currently being generated. Once training is complete, pre-computed Matplotlib PNG comparisons will be dynamically loaded here to showcase the residual error between the Ground Truth CFD and the FNO Surrogate.")
    
    # Render an empty placeholder box so it doesn't look completely blank
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.text(0.5, 0.5, "Awaiting test scenario plots...", ha='center', va='center', fontsize=14, color='gray')
    ax.axis('off')
    st.pyplot(fig)