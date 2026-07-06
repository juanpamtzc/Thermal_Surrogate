import os
import shutil
import subprocess

# --- 1. DIRECTORY SETUP ---
# Use absolute paths so the script can be run from anywhere
PROJECT_ROOT = os.path.expanduser("~/Projects/Thermal_Surrogate")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates", "step9_final_setup")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs/mesh_convergence")

# Ensure the runs directory exists
os.makedirs(RUNS_DIR, exist_ok=True)

# --- 2. DEFINE THE STUDY PARAMETERS ---
# We define a base dictionary of everything that COULD be parameterized.
base_params = {
    "_INLET_VELOCITY_": "0.15",  
    
    # Static Fin Geometry Parameters (Updated to match your actual placeholders)
    "_FIN_START_X_": "0.00625",
    "_FIN_END_X_":   "0.12125",
    "_FIN1_Y_BOTTOM_":  "-0.004075", # Changed from _MIN_ to _BOTTOM_
    "_FIN1_Y_TOP_":     "-0.002825", # Changed from _MAX_ to _TOP_
    "_FIN2_Y_BOTTOM_":  "-0.000625", # Changed from _MIN_ to _BOTTOM_
    "_FIN2_Y_TOP_":     "0.000625",  # Changed from _MAX_ to _TOP_
    "_FIN3_Y_BOTTOM_":  "0.002825",  # Changed from _MIN_ to _BOTTOM_
    "_FIN3_Y_TOP_":     "0.004075"   # Changed from _MAX_ to _TOP_
}

# 2D Mesh Resolutions (perfectly square cells)
# Coarse (1.0x): 0.100 mm cells
# Medium (1.5x): 0.066 mm cells 
# Fine   (2.0x): 0.050 mm cells
mesh_resolutions = {
    "coarse": {"_NX_": "2212", "_NY_": "82"},
    "medium": {"_NX_": "3318", "_NY_": "123"},
    "fine":   {"_NX_": "4424", "_NY_": "164"}
}

# --- 3. EXECUTION LOOP ---
for mesh_name, mesh_params in mesh_resolutions.items():
    case_name = f"mesh_study_{mesh_name}"
    case_path = os.path.join(RUNS_DIR, case_name)
    
    print(f"--- Preparing {case_name} ---")
    
    # 3a. Create a fresh copy of the template
    if os.path.exists(case_path):
        print(f"Removing existing directory: {case_name}")
        shutil.rmtree(case_path)
    
    shutil.copytree(TEMPLATE_DIR, case_path)
    
    # 3b. Combine base parameters with the specific mesh parameters for this run
    current_params = base_params.copy()
    current_params.update(mesh_params)
    
    # 3c. Inject parameters into OpenFOAM files
    # Define which files have placeholders that need replacing
    files_to_modify = [
        os.path.join(case_path, "system", "blockMeshDict"),
        os.path.join(case_path, "0", "fluid", "U"),
        os.path.join(case_path, "system", "topoSetDict") # <--- ADDED THIS LINE
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
    print(f"Starting OpenFOAM for {case_name} (using 8 cores)...")
    
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
        break # Stop the loop if a simulation crashes, rather than wasting time on the next one

print("Mesh convergence study pipeline complete.")