from Method3_Sethian_Saye import startRunMethod3
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


def get_midpoint_path(i): 
    return os.path.join(Base_path, "positions")

def get_phasefield_path(i): 
    return os.path.join(Base_path, "phasedata")

def get_dirty_vertices_path(i): 
    return os.path.join(Base_path, "vertices_not_cleaned_NEW_4")

def get_output_clean_vertices_path(i): 
    return ""

def run_Method3(i_simulation):
    # directory to file that holds all midpoints of simulation i 
    midpoints_path = get_midpoint_path(i_simulation)
    # directory to file that holds all phase fields of simulation i
    phasefield_path = get_phasefield_path(i_simulation)
    # directory for saving dirty vertices of simulation i 
    dirty_vertices_path = get_dirty_vertices_path(i_simulation) 

    startRunMethod3(midpoints_path, phasefield_path, dirty_vertices_path)

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
        executor.map(run_Method3, range(N_Simulations))

    # RUN CLEANING N_Simulations TIMES
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks to the pool
        executor.map(run_Method3, range(N_Simulations))
        






