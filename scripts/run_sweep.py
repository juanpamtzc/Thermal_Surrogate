import os
import subprocess
import re

# 1. Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CLONE_SCRIPT = os.path.join(SCRIPT_DIR, 'clone_case.sh')

# 2. Define our parameter sweep (Heat values in W/m^3)
# We will do 3 test cases: 500kW, 1.5MW, and 2.0MW
HEAT_VALUES = [500000, 1500000, 2000000]

def modify_heat_source(run_id, new_heat_value):
    """Finds the heat value in fvOptions and replaces it."""
    fv_options_path = os.path.join(PROJECT_ROOT, 'openfoam', 'runs', run_id, 'constant', 'fvOptions')
    
    with open(fv_options_path, 'r') as f:
        content = f.read()
    
    # Use Regex to find "explicit <any number>;" and replace it with our new value
    new_content = re.sub(
        r'explicit\s+[0-9]+;', 
        f'explicit    {new_heat_value};', 
        content
    )
    
    with open(fv_options_path, 'w') as f:
        f.write(new_content)
    print(f"  -> Injected new heat value: {new_heat_value} W/m^3")

def run_openfoam_docker(run_id):
    """Executes the Docker container for the specific run."""
    run_dir = os.path.join(PROJECT_ROOT, 'openfoam', 'runs', run_id)
    print(f"  -> Launching OpenFOAM via Docker...")
    
    uid = os.getuid()
    gid = os.getgid()
    
    docker_cmd = [
        "docker", "run", "--rm",
        "-u", f"{uid}:{gid}",
        "-v", f"{run_dir}:/case",
        "-w", "/case",
        "--entrypoint", "/bin/bash",
        "openfoam/openfoam10-paraview510",
        "-c", "source /opt/openfoam10/etc/bashrc && buoyantFoam"
    ]
    
    # Run Docker and save the terminal output to a log file instead of spamming your screen
    log_path = os.path.join(run_dir, "solver.log")
    with open(log_path, 'w') as log_file:
        subprocess.run(docker_cmd, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"  -> Finished! Log saved to solver.log")

if __name__ == "__main__":
    print("🚀 Starting OpenFOAM Parameter Sweep...")
    
    for i, heat_val in enumerate(HEAT_VALUES):
        # Name them run_002, run_003, run_004 (since we already have run_001)
        run_id = f"run_{i+2:03d}" 
        print(f"\n=== Processing {run_id} ===")
        
        # Step A: Clone the pristine template
        subprocess.run([CLONE_SCRIPT, run_id], check=True)
        
        # Step B: Inject the new physics parameter
        modify_heat_source(run_id, heat_val)
        
        # Step C: Run the simulation
        run_openfoam_docker(run_id)
        
    print("\n✅ Sweep Complete! Your synthetic data is ready.")