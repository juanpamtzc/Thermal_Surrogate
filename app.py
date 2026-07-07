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
# 1. Model Statistics
# ---------------------------------------------------------
STATS = {
    't_min': 293.15, 't_max': 380.00,  
    'u_min': 0.1,    'u_max': 0.6,
    'x_min': 0.0,    'x_max': 0.2,
    'y_min': -0.05,  
    'y_max': 0.05,
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
    """
    Synthesizes the 4-channel input exactly matching your true 80x2250 dataset structure.
    """
    # 1. Regenerate coordinate meshes using your TRUE dimensions (80 rows, 2250 columns)
    x = np.linspace(STATS['x_min'], STATS['x_max'], 2250)
    y = np.linspace(STATS['y_min'], STATS['y_max'], 80)
    grid_x, grid_y = np.meshgrid(x, y)  # Outputs shape: (80, 2250)
    
    # 2. Build the Geometry Mask dynamically on the high-res grid
    geom_mask = np.ones((80, 2250))
    fin_thickness = 0.02  
    
    # Solid domain boundary calculation
    y_lower = offset - (fin_thickness / 2.0)
    y_upper = offset + (fin_thickness / 2.0)
    
    # Mark pixels inside the solid fin as 0
    mask_condition = (grid_x <= length) & (grid_y >= y_lower) & (grid_y <= y_upper)
    geom_mask[mask_condition] = 0.0
                    
    # 3. Create the input velocity array matching shape (80, 2250)
    u_inlet = np.ones((80, 2250)) * vel
    
    # 4. Apply Min-Max Normalization exactly matching your dataset.py
    u_inlet_norm = (u_inlet - STATS['u_min']) / (STATS['u_max'] - STATS['u_min'] + 1e-8)
    grid_x_norm = (grid_x - STATS['x_min']) / (STATS['x_max'] - STATS['x_min'] + 1e-8)
    grid_y_norm = (grid_y - STATS['y_min']) / (STATS['y_max'] - STATS['y_min'] + 1e-8)
    
    # 5. Collate, stack channels, and process tensor shapes
    input_channels = [
        geom_mask.astype(np.float32),
        u_inlet_norm.astype(np.float32),
        grid_x_norm.astype(np.float32),
        grid_y_norm.astype(np.float32)
    ]
    X_array = np.stack(input_channels, axis=0)
    X_tensor = torch.from_numpy(X_array).float().unsqueeze(0).to(device)
    
    # 6. Model Evaluation
    with torch.no_grad():
        pred_T_norm = model(X_tensor).squeeze().cpu().numpy()
        
    # 7. Un-normalize output tensor back to Kelvin physical scale
    pred_T = pred_T_norm * (STATS['t_max'] - STATS['t_min']) + STATS['t_min']
    
    # 8. Force solid structure bounds to 0.0K to match your notebook's dead zones
    pred_T[geom_mask == 0.0] = 0.0 
    
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
        offset = st.slider("Fin Y-Center Offset (m)", -0.015, 0.015, 0.0, step=0.0005)

    with col2:
        st.subheader("FNO Predicted Temperature Field")
        if model is not None:
            pred_T = run_interactive_inference(vel, length, offset)
            
            # --- Notebook Robust Visual Percentile Bound Scaling ---
            valid_mask = pred_T > 100  # Ignore the 0.0K dead zones for visual scaling
            if np.any(valid_mask):
                vmin = np.percentile(pred_T[valid_mask], 1)
                vmax = np.percentile(pred_T[valid_mask], 99)
            else:
                vmin, vmax = 297.0, 360.0
            
            fig, ax = plt.subplots(figsize=(15, 4))
            
            # Matches your notebook's clean matrix coordinate plotting approach
            im = ax.imshow(
                pred_T, 
                cmap='inferno', 
                aspect='auto', 
                origin='lower',
                vmin=vmin, 
                vmax=vmax
            )
            
            ax.set_title(f"FNO AI Prediction | Inlet Velocity: {vel:.2f} m/s", fontsize=12)
            fig.colorbar(im, ax=ax, fraction=0.015, pad=0.04, label="Temp (K)")
            
            plt.tight_layout()
            st.pyplot(fig)

with tab2:
    st.subheader("Model Validation against Ground Truth CFD")
    st.info("🚧 **Simulations Running** 🚧 \n\nA new OpenFOAM dataset is currently being generated.")