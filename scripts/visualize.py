import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure Python can find your 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ThermalFNO

def get_train_stats(train_dir):
    """Scans the training data to get the exact min/max bounds used for normalization."""
    print("Scanning training set for normalization bounds...")
    train_cases = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    t_mins, t_maxs = [], []
    u_mins, u_maxs = [], []
    x_mins, x_maxs = [], []
    y_mins, y_maxs = [], []

    for case in train_cases:
        t_mins.append(np.min(np.load(os.path.join(case, "temperature.npy"))))
        t_maxs.append(np.max(np.load(os.path.join(case, "temperature.npy"))))
        u_mins.append(np.min(np.load(os.path.join(case, "u_inlet.npy"))))
        u_maxs.append(np.max(np.load(os.path.join(case, "u_inlet.npy"))))
        x_mins.append(np.min(np.load(os.path.join(case, "grid_x.npy"))))
        x_maxs.append(np.max(np.load(os.path.join(case, "grid_x.npy"))))
        y_mins.append(np.min(np.load(os.path.join(case, "grid_y.npy"))))
        y_maxs.append(np.max(np.load(os.path.join(case, "grid_y.npy"))))

    return {
        't_min': np.min(t_mins), 't_max': np.max(t_maxs),
        'u_min': np.min(u_mins), 'u_max': np.max(u_maxs),
        'x_min': np.min(x_mins), 'x_max': np.max(x_maxs),
        'y_min': np.min(y_mins), 'y_max': np.max(y_maxs),
    }

def visualize_test_case():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
    TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fno_best.pt")
    PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. Get normalization stats & load model
    stats = get_train_stats(TRAIN_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalFNO(modes1=12, modes2=32, width=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Pick the first test case
    test_cases = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    target_case = os.path.join(TEST_DIR, test_cases[0])
    print(f"\nVisualizing Test Case: {test_cases[0]}")

    # Load raw numpy arrays
    geom_mask = np.load(os.path.join(target_case, "geom_mask.npy")) 
    u_inlet = np.load(os.path.join(target_case, "u_inlet.npy"))     
    grid_x = np.load(os.path.join(target_case, "grid_x.npy"))      
    grid_y = np.load(os.path.join(target_case, "grid_y.npy"))      
    true_T = np.load(os.path.join(target_case, "temperature.npy")) 

    # 3. Normalize inputs perfectly matching the training loop
    u_inlet_norm = (u_inlet - stats['u_min']) / (stats['u_max'] - stats['u_min'] + 1e-8)
    grid_x_norm = (grid_x - stats['x_min']) / (stats['x_max'] - stats['x_min'] + 1e-8)
    grid_y_norm = (grid_y - stats['y_min']) / (stats['y_max'] - stats['y_min'] + 1e-8)

    input_channels = np.stack([geom_mask, u_inlet_norm, grid_x_norm, grid_y_norm], axis=0)
    X_tensor = torch.from_numpy(input_channels).float().unsqueeze(0).to(device)

    # 4. Run FNO Prediction
    with torch.no_grad():
        pred_T_norm = model(X_tensor).squeeze().cpu().numpy()

    # 5. Un-normalize the prediction back to real physical temperatures
    pred_T = pred_T_norm * (stats['t_max'] - stats['t_min']) + stats['t_min']
    
    # Calculate absolute error
    error = np.abs(true_T - pred_T)

    # 6. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
    
    # Define color limits so both plots share the exact same scale
    vmin, vmax = np.min(true_T), np.max(true_T)

    # Plot 1: OpenFOAM Ground Truth
    im1 = axes[0].imshow(true_T, cmap='inferno', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"OpenFOAM CFD (Ground Truth)\nInlet Velocity: {u_inlet[0,0]:.2f} m/s", fontsize=12)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Temp (K)")

    # Plot 2: AI Prediction
    im2 = axes[1].imshow(pred_T, cmap='inferno', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("FNO AI Prediction", fontsize=12)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Temp (K)")

    # Plot 3: Absolute Error Map
    im3 = axes[2].imshow(error, cmap='Reds', aspect='auto', origin='lower')
    axes[2].set_title(f"Absolute Error (Max Error: {np.max(error):.2f} K)", fontsize=12)
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="Error (K)")

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "fno_vs_openfoam.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved successfully to: {save_path}")

if __name__ == "__main__":
    visualize_test_case()