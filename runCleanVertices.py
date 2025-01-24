from Method3_Sethian_Saye import startRun 
from clean_vertices_and_calculate_pAtics import startCleaning

def get_midpoint_path(i): 
    return ""

def get_phasefield_path(i): 
    return ""

def get_dirty_vertices_path(i): 
    return ""

def get_output_clean_vertices_path(i): 
    return ""

# TODO: check N_Simulations
N_Simulations = 420
for i in range(N_Simulations):
    # directory to file that holds all midpoints of simulation i 
    midpoints_path = get_midpoint_path(i)
    # directory to file that holds all phase fields of simulation i
    phasefield_path = get_phasefield_path(i)
    # directory for saving dirty vertices of simulation i 
    dirty_vertices_path = get_dirty_vertices_path(i) 

    startRun(midpoints_path, phasefield_path, dirty_vertices_path)

for i in range(N_Simulations):
    dirty_vertices_path = get_dirty_vertices_path(i) 
    output_clean_vertices_path = get_output_clean_vertices_path(i)
    startCleaning(dirty_vertices_path, output_clean_vertices_path)






