import os
import glob
import numpy as np
import pyvista as pv
from tqdm import tqdm

def extract_case_to_numpy(case_path, nx=2212, ny=82):
    """Reads a completed OpenFOAM case and saves 2D arrays for the ML Dataset."""
    
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
    
    # --- THE ROBUST GEOMETRIC MAPPING ---
    # We use the exact physical bounding box from blockMesh
    x_min, x_max = 0.0, 0.22125
    y_min, y_max = -0.004075, 0.004075
    
    # Calculate the exact array index [0 to nx-1] for every cell
    x_indices = np.floor((coords[:, 0] - x_min) / (x_max - x_min) * nx).astype(int)
    y_indices = np.floor((coords[:, 1] - y_min) / (y_max - y_min) * ny).astype(int)
    
    # Clip to bounds just to be safe from floating-point edge cases
    x_indices = np.clip(x_indices, 0, nx - 1)
    y_indices = np.clip(y_indices, 0, ny - 1)
    
    # Initialize empty 2D tensors
    T_grid = np.zeros((ny, nx))
    U_grid = np.zeros((ny, nx))
    Mask_grid = np.zeros((ny, nx))
    
    # Generate the coordinate grid mathematically
    x_1d = np.linspace(x_min + (x_max-x_min)/(2*nx), x_max - (x_max-x_min)/(2*nx), nx)
    y_1d = np.linspace(y_min + (y_max-y_min)/(2*ny), y_max - (y_max-y_min)/(2*ny), ny)
    X_grid, Y_grid = np.meshgrid(x_1d, y_1d)
    
    # Map the unstructured OpenFOAM data onto the structured 2D grid
    T_grid[y_indices, x_indices] = T_all
    U_grid[y_indices, x_indices] = U_all
    Mask_grid[y_indices, x_indices] = Mask_all
    
    np.save(os.path.join(case_path, "temperature.npy"), T_grid)
    np.save(os.path.join(case_path, "u_inlet.npy"), U_grid)
    np.save(os.path.join(case_path, "geom_mask.npy"), Mask_grid)
    np.save(os.path.join(case_path, "grid_x.npy"), X_grid)
    np.save(os.path.join(case_path, "grid_y.npy"), Y_grid)
    
    return True

if __name__ == "__main__":
    base_dir = os.path.expanduser("~/Projects/Thermal_Surrogate/data")
    
    for split in ["train", "test"]:
        print(f"\nExtracting {split.upper()} data...")
        cases = glob.glob(os.path.join(base_dir, split, f"{split}_*"))
        
        for case in tqdm(cases):
            extract_case_to_numpy(case)
            
    print("\n✅ All OpenFOAM cases extracted to ML Tensors!")