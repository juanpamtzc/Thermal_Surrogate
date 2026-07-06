import os
import numpy as np
import pandas as pd
import pyvista as pv
from typing import Dict, List, Tuple

def extract_patch_data(foam_file: str, patch_names: Dict[str, str], time_step: float = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extracts spatial coordinates, temperatures, and heat fluxes from OpenFOAM patches.
    """
    if not os.path.exists(foam_file):
        raise FileNotFoundError(f"Could not find OpenFOAM file: {foam_file}")
        
    reader = pv.OpenFOAMReader(foam_file)
    reader.enable_all_cell_arrays()
    
    if time_step is None:
        reader.set_active_time_value(reader.time_values[-1])
    else:
        reader.set_active_time_value(time_step)
        
    mesh = reader.read()
    
    # Safety check for the fluid region
    if "fluid" not in mesh.keys():
        raise KeyError(f"Region 'fluid' not found. Available regions: {mesh.keys()}")
        
    fluid_boundaries = mesh["fluid"]["boundary"]
    
    results = {}
    for label, patch_name in patch_names.items():
        # --- ROBUST ERROR HANDLING ---
        if patch_name not in fluid_boundaries.keys():
            available_patches = list(fluid_boundaries.keys())
            raise KeyError(
                f"\n🚨 Patch '{patch_name}' not found!\n"
                f"Available patches in the fluid region are:\n{available_patches}\n"
                f"Please update the 'patches' dictionary in your notebook to match these exactly."
            )
            
        wall_mesh_points = fluid_boundaries[patch_name].cell_data_to_point_data()
        
        # Safety check for required arrays
        if "T" not in wall_mesh_points.array_names or "wallHeatFlux" not in wall_mesh_points.array_names:
             raise ValueError(
                 f"Missing required arrays on patch '{patch_name}'. "
                 f"Found arrays: {wall_mesh_points.array_names}. "
                 f"Ensure 'T' and 'wallHeatFlux' are being calculated by OpenFOAM."
             )
        
        results[label] = {
            'X': wall_mesh_points.points[:, 0],
            'T': wall_mesh_points["T"],
            'q_mag': np.abs(wall_mesh_points["wallHeatFlux"])
        }
        
    return results

def load_convergence_log(log_file: str) -> pd.DataFrame:
    """Safely loads an OpenFOAM postProcessing log file into a DataFrame."""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
        
    df = pd.read_csv(log_file, comment='#', sep='\t', skipinitialspace=True, header=None)
    df.columns = ['Time', 'Value']
    return df

def check_convergence(series: np.ndarray, window: int = 100, tolerance: float = 1e-4) -> bool:
    """Evaluates relative variation over the last 'window' iterations."""
    if len(series) < window:
        return False
        
    recent_history = np.array(series[-window:])
    max_val, min_val, mean_val = np.max(recent_history), np.min(recent_history), np.mean(recent_history)
    
    if abs(mean_val) < 1e-12:
        relative_variation = max_val - min_val
    else:
        relative_variation = abs((max_val - min_val) / mean_val)
        
    return relative_variation <= tolerance