import os
import sys
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ThermalFNO

def generate_validation_plots():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
    
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # 1. Load Stats
    with open(os.path.join(MODEL_DIR, "normalization_stats.json"), "r") as f:
        stats = json.load(f)

    # 2. Load Model
    device = torch.device("cpu")
    model = ThermalFNO(modes1=12, modes2=32, width=32).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "fno_best.pt"), map_location=device))
    model.eval()

    # 3. Grab and sort test cases by Inlet Velocity to guarantee diversity
    required_files = ["temperature.npy", "u_converged.npy", "u_inlet.npy", "geom_mask.npy", "grid_x.npy", "grid_y.npy"]
    valid_cases = []
    
    for d in os.listdir(TEST_DIR):
        case_path = os.path.join(TEST_DIR, d)
        if os.path.isdir(case_path) and all(os.path.exists(os.path.join(case_path, f)) for f in required_files):
            try:
                # Parse velocity from folder name
                vel = float(d.split('_U')[1].split('_L')[0].replace('p', '.'))
                valid_cases.append((vel, case_path))
            except Exception:
                continue

    if len(valid_cases) < 3:
        print("Not enough valid test cases found to generate 3 diverse plots!")
        return

    valid_cases.sort(key=lambda x: x[0])
    
    # Pick Lowest, Median, and Highest velocity cases
    test_cases = [
        valid_cases[0][1],                  
        valid_cases[len(valid_cases)//2][1], 
        valid_cases[-1][1]                  
    ]

    physical_extent = [0.0, 0.22125, -0.004075, 0.004075]

    for i, case_path in enumerate(test_cases):
        case_name = os.path.basename(case_path)
        print(f"Plotting diverse case {i+1}/3: {case_name}...")

        # Load raw arrays
        geom_mask = np.load(os.path.join(case_path, "geom_mask.npy"))
        u_inlet = np.load(os.path.join(case_path, "u_inlet.npy"))
        grid_x = np.load(os.path.join(case_path, "grid_x.npy"))
        grid_y = np.load(os.path.join(case_path, "grid_y.npy"))
        true_T = np.load(os.path.join(case_path, "temperature.npy"))
        true_U = np.load(os.path.join(case_path, "u_converged.npy"))

        # Normalize Inputs
        u_inlet_norm = (u_inlet - stats['ui_min']) / (stats['ui_max'] - stats['ui_min'] + 1e-8)
        grid_x_norm = (grid_x - stats['x_min']) / (stats['x_max'] - stats['x_min'] + 1e-8)
        grid_y_norm = (grid_y - stats['y_min']) / (stats['y_max'] - stats['y_min'] + 1e-8)

        # Predict
        X_array = np.stack([geom_mask, u_inlet_norm, grid_x_norm, grid_y_norm], axis=0).astype(np.float32)
        X_tensor = torch.from_numpy(X_array).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(X_tensor).squeeze().numpy()
        
        # Un-normalize
        pred_T = pred_norm[0] * (stats['t_max'] - stats['t_min']) + stats['t_min']
        pred_U = pred_norm[1] * (stats['uc_max'] - stats['uc_min']) + stats['uc_min']
        
        # Force physical velocity constraint (solid U = 0), leave T untouched
        pred_U[geom_mask == 0.0] = 0.0 
        
        # Calculate Error & Metrics
        error_T = np.abs(true_T - pred_T)
        error_U = np.abs(true_U - pred_U)
        
        fluid_mae_T = np.mean(error_T[geom_mask == 1.0])
        solid_mae_T = np.mean(error_T[geom_mask == 0.0])
        fluid_mae_U = np.mean(error_U[geom_mask == 1.0])
        # Note: We do not report solid MAE for Velocity because solid U is physically constrained to exactly 0.0 m/s in CHT.

        # Dynamic max error (99.5th percentile)
        vmax_err_T = np.percentile(error_T, 99.5) + 1e-3
        vmax_err_U = np.percentile(error_U, 99.5) + 1e-3

        # --- PLOTTING (3x2 Grid for Dual Physics) ---
        fig, axes = plt.subplots(3, 2, figsize=(18, 10), gridspec_kw={'hspace': 0.4, 'wspace': 0.1})
        
        # Helper to set labels
        for ax in axes.flat:
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

        # Row 1: Ground Truth
        im0_T = axes[0, 0].imshow(true_T, cmap='inferno', aspect='auto', origin='lower', extent=physical_extent, vmin=stats['t_min'], vmax=stats['t_max'])
        axes[0, 0].set_title(f"True Temp (K)")
        fig.colorbar(im0_T, ax=axes[0, 0], pad=0.01, label="Temp (K)")

        im0_U = axes[0, 1].imshow(true_U, cmap='viridis', aspect='auto', origin='lower', extent=physical_extent, vmin=stats['uc_min'], vmax=stats['uc_max'])
        axes[0, 1].set_title(f"True Velocity (m/s)")
        fig.colorbar(im0_U, ax=axes[0, 1], pad=0.01, label="Velocity (m/s)")

        # Row 2: FNO Prediction
        im1_T = axes[1, 0].imshow(pred_T, cmap='inferno', aspect='auto', origin='lower', extent=physical_extent, vmin=stats['t_min'], vmax=stats['t_max'])
        axes[1, 0].set_title("FNO Predicted Temp")
        fig.colorbar(im1_T, ax=axes[1, 0], pad=0.01, label="Temp (K)")

        im1_U = axes[1, 1].imshow(pred_U, cmap='viridis', aspect='auto', origin='lower', extent=physical_extent, vmin=stats['uc_min'], vmax=stats['uc_max'])
        axes[1, 1].set_title("FNO Predicted Velocity")
        fig.colorbar(im1_U, ax=axes[1, 1], pad=0.01, label="Velocity (m/s)")

        # Row 3: Absolute Error with explicit regional MAE
        im2_T = axes[2, 0].imshow(error_T, cmap='magma', aspect='auto', origin='lower', extent=physical_extent, vmin=0, vmax=vmax_err_T)
        axes[2, 0].set_title(f"Abs Error T | Fluid MAE: {fluid_mae_T:.2f} K | Solid MAE: {solid_mae_T:.2f} K")
        fig.colorbar(im2_T, ax=axes[2, 0], pad=0.01, label="Error (K)")

        im2_U = axes[2, 1].imshow(error_U, cmap='magma', aspect='auto', origin='lower', extent=physical_extent, vmin=0, vmax=vmax_err_U)
        axes[2, 1].set_title(f"Abs Error U | Fluid MAE: {fluid_mae_U:.3f} m/s")
        fig.colorbar(im2_U, ax=axes[2, 1], pad=0.01, label="Error (m/s)")

        plt.suptitle(f"Holdout Validation: {case_name}", fontsize=16)
        plt.savefig(os.path.join(ASSETS_DIR, f"val_case_{i+1}.png"), bbox_inches='tight', dpi=150)
        plt.close()

    print(f"✅ Generated dual-physics validation images in {ASSETS_DIR}/")

if __name__ == "__main__":
    generate_validation_plots()