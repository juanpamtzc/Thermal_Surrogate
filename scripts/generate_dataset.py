import os
import shutil
import subprocess
import numpy as np
from scipy.stats.qmc import LatinHypercube

# --- 1. DIRECTORY SETUP ---
PROJECT_ROOT = os.path.expanduser("~/Projects/Thermal_Surrogate")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates", "step9_final_setup")

TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# --- 2. DEFINE THE PARAMETER SPACE ---
# Parameters: [Velocity (m/s), Fin Length (m), Fin 2 Y-Center (m)]
# Velocity: 0.5 to 5.0 m/s
# Length: 0.05 to 0.15 m
# Fin 2 Y-Center: -0.0015 to 0.0015 m (limits movement so it doesn't collide with Fin 1 or 3)
bounds_min = [0.1, 0.01, -0.0015] 
bounds_max = [0.6, 0.115,  0.0015] 

N_TRAIN = 250
N_TEST = 50
TOTAL_SAMPLES = N_TRAIN + N_TEST

# Generate LHS Samples for the ENTIRE dataset at once
print(f"Generating Latin Hypercube Samples for {TOTAL_SAMPLES} points...")
sampler = LatinHypercube(d=3, seed=42)
sample_points = sampler.random(n=TOTAL_SAMPLES)

# Scale samples to physical bounds
scaled_samples = bounds_min + sample_points * (np.array(bounds_max) - np.array(bounds_min))

# Split into train and test sets
train_samples = scaled_samples[:N_TRAIN]
test_samples = scaled_samples[N_TRAIN:]

# --- 3. HELPER FUNCTION TO RUN SIMULATIONS ---
def run_batch(samples, output_dir, prefix):
    # Fin 2 thickness is 0.00125 m (from original topoSetDict)
    FIN2_HALF_THICKNESS = 0.000625
    FIN_START_X = 0.00625
    
    for i, (vel, length, y_center) in enumerate(samples):
        # Snap length and y_center to nearest 0.1 mm (0.0001 m) to align with cell centers
        length_snapped = round(length / 0.0001) * 0.0001
        y_center_snapped = round(y_center / 0.0001) * 0.0001
        
        # Calculate derived geometry bounds
        fin_end_x = FIN_START_X + length_snapped
        fin2_y_bottom = y_center_snapped - FIN2_HALF_THICKNESS
        fin2_y_top = y_center_snapped + FIN2_HALF_THICKNESS
        
        # Format names cleanly: e.g., train_001_U2p50_L0p120_Y0p0010
        case_name = f"{prefix}_{i:03d}_U{vel:.2f}_L{length_snapped:.3f}_Y{y_center_snapped:.4f}".replace('.', 'p').replace('-', 'm')
        case_path = os.path.join(output_dir, case_name)
        
        print(f"--- Preparing {case_name} ---")
        
        if os.path.exists(case_path):
            shutil.rmtree(case_path)
        shutil.copytree(TEMPLATE_DIR, case_path)
        
        current_params = {
            "_INLET_VELOCITY_": f"{vel:.3f}",
            "_FIN_END_X_": f"{fin_end_x:.6f}",
            "_FIN2_Y_BOTTOM_": f"{fin2_y_bottom:.6f}",
            "_FIN2_Y_TOP_": f"{fin2_y_top:.6f}",
            "_NX_": "2212",  # Injecting our fixed mesh resolution
            "_NY_": "82"
        }
        
        # Inject parameters into blockMeshDict, topoSetDict, and U
        files_to_modify = [
            os.path.join(case_path, "system", "blockMeshDict"), # Added this back!
            os.path.join(case_path, "system", "topoSetDict"),
            os.path.join(case_path, "0", "fluid", "U")
        ]
        
        for filepath in files_to_modify:
            if os.path.exists(filepath):
                with open(filepath, 'r') as file: file_data = file.read()
                for placeholder, value in current_params.items():
                    file_data = file_data.replace(placeholder, str(value))
                with open(filepath, 'w') as file: file.write(file_data)
            else:
                print(f"⚠️ Warning: Could not find {filepath}")

        # Execute Docker command
        docker_cmd = (
            f"docker run --rm -u $(id -u):$(id -g) "
            f"-v {case_path}:/case -w /case --entrypoint /bin/bash "
            f"openfoam/openfoam10-paraview510 -c "
            f"\"source /opt/openfoam10/etc/bashrc && chmod +x ./Allrun && ./Allrun\""
        )
        
        try:
            # Output is hidden to keep terminal clean; check the case directory if it fails
            subprocess.run(docker_cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
            print(f"✅ Finished {case_name}")
        except subprocess.CalledProcessError:
            print(f"❌ Error executing {case_name}. Moving to next.")

# --- 4. EXECUTE BATCHES ---
print("\n🚀 STARTING TRAINING DATA GENERATION")
run_batch(train_samples, TRAIN_DIR, "train")

print("\n🚀 STARTING TESTING DATA GENERATION")
run_batch(test_samples, TEST_DIR, "test")

print("\n🎉 Full dataset generation complete!")