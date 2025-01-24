from Method3_Sethian_Saye import startRun 

def get_midpoint_path(i): 
    return ""

def get_phasefield_path(i): 
    return ""

def get_output_path(i): 
    return ""

# TODO: check N_Simulations
N_Simulations = 420
for i in range(N_Simulations):
    # directory to file that holds all midpoints of simulation i 
    midpoints_path = get_midpoint_path(i)
    # directory to file that holds all phase fields of simulation i
    phasefield_path = get_phasefield_path(i)
    # directory for saving dirty vertices of simulation i 
    output_path = get_output_path(i) 
    startRun(midpoints_path, midpoints_path, midpoints_path, output_path)






