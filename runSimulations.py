from Method3_Sethian_Saye_FOR_LINUX import startRunMethod3
from clean_vertices_and_calculate_pAtics import startCleaning
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import sys

home = Path.home()

# path to code directory
Code_path = os.path.join(home, "Onedrive", "Desktop", "researchProject", "code")

# path to VTK suite:
VTK_path = os.path.join(Code_path, "vtk-suite")
sys.path.append(VTK_path)

# Path to Output directory 
Output_path = os.path.join(Code_path, "output")

# path for input file 
Base_path = os.path.join(Code_path, "o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1")

# path to uncleaned vertices directory 
Vertices_Path = os.path.join(Base_path, "vertices_not_cleaned_NEW_3")

Core_Path = os.path.join("data", "PAticsProject", "Daten_Harish_Jain")
Ca_directories = ["Ca0", "Ca1", "Ca2", "Ca3"]
absolute_Ca0 = os.path.join(Core_Path, "Ca0")
# relative dir names 
o2024dirs = [d.name for d in absolute_Ca0.iterdir() if d.is_dir()]

def run_Method3(time):

    global Core_Path
    # time_steps = [75.0, 75.5, 76.0, ... , 150.0] 
    for Ca_directory in Ca_directories:
        for simulation in o2024dirs:
            midpoints_path = os.path.join(Core_Path, Ca_directory, simulation, "positions")
            phasefield_path = os.path.join(Core_Path, Ca_directory, simulation, "data")
            # TODO: this dir must be created on linux pc
            dirty_vertices_path = os.path.join(Core_Path, Ca_directory, simulation, "dirty_vertices")
            startRunMethod3(midpoints_path, phasefield_path, dirty_vertices_path, time)

def run_Cleaning(i_simulation):
    dirty_vertices_path = get_dirty_vertices_path(i_simulation) 
    output_clean_vertices_path = get_output_clean_vertices_path(i_simulation)
    startCleaning(dirty_vertices_path, output_clean_vertices_path)


# TODO: check N_Simulations
N_Simulations = 1

if __name__ == "__main__":
    # Total number of simulations
    N_Simulations = 1

    
    # RUN METHOD3 N_Simulations TIMES parralellized 
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks to the pool
        time_steps = [x/2 for x in range(150, 301)]
        executor.map(run_Method3, time_steps)

    # RUN CLEANING N_Simulations TIMES
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks to the pool
        executor.map(run_Method3, range(N_Simulations))
        






