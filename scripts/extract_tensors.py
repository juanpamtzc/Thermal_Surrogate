import os
import glob
import re
import numpy as np
import pyvista as pv
from tqdm import tqdm

def get_true_inlet_velocity(case_path):
    """
    Parses the actual OpenFOAM boundary condition file for the true inlet velocity,
    with a cross-check against the filename to prevent silent data corruption.
    """
    case_name = os.path.basename(os.path.normpath(case_path))
    u_file = os.path.join(case_path, "0", "fluid", "U")
    
    if not os.path.exists(u_file):
        raise FileNotFoundError(f"Could not find velocity file: {u_file}")
        
    with open(u_file, 'r') as f:
        content = f.read()
        
    # --- 1. Parse from File ---
    # Matches macro definition: Uinlet (0.350 0 0); 
    match_macro = re.search(r'Uinlet\s+\(\s*([-\d.eE]+)\s+[-\d.eE]+\s+[-\d.eE]+\s*\)', content)
    # Matches direct assignment: value uniform (0.350 0 0); inside the inlet block
    match_direct = re.search(r'inlet\s*\{[^}]*value\s+uniform\s+\(\s*([-\d.eE]+)\s+[-\d.eE]+\s+[-\d.eE]+\s*\)', content, re.MULTILINE | re.DOTALL)
    
    if match_macro:
        file_vel = float(match_macro.group(1))
    elif match_direct:
        file_vel = float(match_direct.group(1))
    else:
        raise ValueError(f"Could not parse inlet velocity from {u_file}")

    # --- 2. Parse from Filename (Cross-check) ---
    try:
        vel_str = case_name.split('_U')[1].split('_L')[0]
        filename_vel = float(vel_str.replace('p', '.'))
    except Exception as e:
        raise ValueError(f"Could not parse velocity from filename {case_name}: {e}")

    # --- 3. Safety Assertion (Tolerance 0.01 to account for string rounding) ---
    assert abs(file_vel - filename_vel) <= 0.01, \
        f"Velocity mismatch in {case_name}! File: {file_vel}, Filename: {filename_vel}"
        
    return file_vel

def extract_case_to_numpy(case_path, nx=2212, ny=82):
    foam_file = os.path.join(case_path, "case.foam")
    if not os.path.exists(foam_file):
        open(foam_file, 'w').close()
        
    reader = pv.OpenFOAMReader(foam_file)
    reader.set_active_time_value(reader.time_values[-1])
    mesh = reader.read()
    
    cell_centers = []
    t_vals, u_vals, mask_vals = [], [], []
    
    for block_name in mesh.keys():
        if block_name == "boundary":
            continue
            
        block = mesh[block_name]["internalMesh"]
        cell_centers.append(block.cell_centers().points)
        
        t_vals.append(block.cell_data.get("T", np.zeros(block.n_cells)))
        
        if "U" in block.cell_data:
            u_vals.append(block.cell_data["U"][:, 0])
        else:
            u_vals.append(np.zeros(block.n_cells))
            
        is_fluid = 1.0 if ("fluid" in block_name.lower() or "region1" in block_name.lower()) else 0.0
        mask_vals.append(np.full(block.n_cells, is_fluid))

    coords = np.vstack(cell_centers)
    T_all = np.concatenate(t_vals)
    U_all = np.concatenate(u_vals)
    Mask_all = np.concatenate(mask_vals)
    
    # --- PRECISE HEX MESH BIJECTION ---
    x_min, x_max = 0.0, 0.22125
    y_min, y_max = -0.004075, 0.004075
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    
    # Use np.round to guarantee 1-to-1 mapping without floating point drift
    x_indices = np.round((coords[:, 0] - x_min - dx/2) / dx).astype(int)
    y_indices = np.round((coords[:, 1] - y_min - dy/2) / dy).astype(int)
    
    x_indices = np.clip(x_indices, 0, nx - 1)
    y_indices = np.clip(y_indices, 0, ny - 1)
    
    T_grid = np.zeros((ny, nx))
    U_converged_grid = np.zeros((ny, nx))
    Mask_grid = np.zeros((ny, nx))
    
    T_grid[y_indices, x_indices] = T_all
    U_converged_grid[y_indices, x_indices] = U_all
    Mask_grid[y_indices, x_indices] = Mask_all

    x_1d = np.linspace(x_min + dx/2, x_max - dx/2, nx)
    y_1d = np.linspace(y_min + dy/2, y_max - dy/2, ny)
    X_grid, Y_grid = np.meshgrid(x_1d, y_1d)
    
    # --- ROBUST VELOCITY ASSIGNMENT ---
    inlet_velocity_scalar = get_true_inlet_velocity(case_path)
    U_inlet_grid = np.full((ny, nx), inlet_velocity_scalar)
    
    # Save Inputs (What the FNO sees)
    np.save(os.path.join(case_path, "geom_mask.npy"), Mask_grid)
    np.save(os.path.join(case_path, "u_inlet.npy"), U_inlet_grid)
    np.save(os.path.join(case_path, "grid_x.npy"), X_grid)
    np.save(os.path.join(case_path, "grid_y.npy"), Y_grid)
    
    # Save Targets (What the FNO predicts)
    np.save(os.path.join(case_path, "temperature.npy"), T_grid)
    np.save(os.path.join(case_path, "u_converged.npy"), U_converged_grid)
    
    return True

if __name__ == "__main__":
    base_dir = os.path.expanduser("~/Projects/Thermal_Surrogate/data")
    
    for split in ["train", "test"]:
        print(f"\nExtracting {split.upper()} data...")
        cases = glob.glob(os.path.join(base_dir, split, f"{split}_*"))
        for case in tqdm(cases):
            # Gracefully handle single-case corruption without crashing the batch
            try:
                extract_case_to_numpy(case)
            except (AssertionError, ValueError, FileNotFoundError) as e:
                print(f"\n⚠️ Skipping {os.path.basename(case)}: {e}")
            
    print("\n✅ All OpenFOAM cases cleanly extracted!")