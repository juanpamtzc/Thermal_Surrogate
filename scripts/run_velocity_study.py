import os
import shutil
import subprocess

# --- 1. DIRECTORY SETUP ---
# Use absolute paths so the script can be run from anywhere
PROJECT_ROOT = os.path.expanduser("~/Projects/Thermal_Surrogate")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates", "step9_final_setup")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# Ensure the runs directory exists
os.makedirs(RUNS_DIR, exist_ok=True)

# --- 2. DEFINE THE STUDY PARAMETERS ---
# We define a base dictionary of the fixed parameters.
# Since we achieved grid independence, we lock in the coarse mesh values here.
base_params = {
    "_NX_": "2212",
    "_NY_": "82",
}

# The velocities we want to sweep (in m/s)
velocities_to_test = ["0.5", "1.0", "2.0", "3.0", "5.0"]

# --- 3. EXECUTION LOOP ---
for vel in velocities_to_test:
    # Replace the decimal with a 'p' for safe folder names (e.g., 2.0 -> 2p0)
    vel_str = vel.replace('.', 'p')
    case_name = f"velocity_sweep_U{vel_str}"
    case_path = os.path.join(RUNS_DIR, case_name)
    
    print(f"--- Preparing {case_name} (Velocity = {vel} m/s) ---")
    
    # 3a. Create a fresh copy of the template
    if os.path.exists(case_path):
        print(f"Removing existing directory: {case_name}")
        shutil.rmtree(case_path)
    
    shutil.copytree(TEMPLATE_DIR, case_path)
    
    # 3b. Combine base parameters with the specific velocity for this run
    current_params = base_params.copy()
    current_params["_INLET_VELOCITY_"] = vel
    
    # 3c. Inject parameters into OpenFOAM files
    # Define which files have placeholders that need replacing
    files_to_modify = [
        os.path.join(case_path, "system", "blockMeshDict"),
        os.path.join(case_path, "0", "fluid", "U") 
    ]
    
    for filepath in files_to_modify:
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                file_data = file.read()
                
            for placeholder, value in current_params.items():
                file_data = file_data.replace(placeholder, str(value))
                
            with open(filepath, 'w') as file:
                file.write(file_data)
        else:
            print(f"Warning: Could not find {filepath} to inject parameters.")

    # 3d. Execute Docker command
    print(f"Starting OpenFOAM for {case_name} (using Docker)...")
    
    # We construct the Docker command dynamically to mount the specific run directory.
    # Using shell=True here because of the complex bash string and subshells $(id -u).
    docker_cmd = (
        f"docker run --rm -u $(id -u):$(id -g) "
        f"-v {case_path}:/case -w /case --entrypoint /bin/bash "
        f"openfoam/openfoam10-paraview510 -c "
        f"\"source /opt/openfoam10/etc/bashrc && chmod +x ./Allrun && ./Allrun\""
    )
    
    try:
        # Run the command and stream the output to the terminal so you can monitor progress
        subprocess.run(docker_cmd, shell=True, check=True)
        print(f"Finished {case_name} successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {case_name}. Simulation failed.")
        print(e)
        break # Stop the loop if a simulation crashes

print("Velocity sweep pipeline complete.")