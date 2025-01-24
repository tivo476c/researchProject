"""
PREREQUISITS: the following directories and files must be saved in the executing system: 
* skfmm must be installed, you will probably need to install visual studio so that the installation of skfmm works
* vtk-suite
* phase field input file "o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1"

THE FOLLOWING PATHS MUST BE SET MANUALLY IN THE NEXT LINES 
"""

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

# path to fine grid directory 
Fine_grid_200x200_path = os.path.join(Code_path, "small_fine_grid_template_200x200.vtu")

# path to fine grid directory 
Fine_grid_600x600_path = os.path.join(Code_path, "small_fine_grid_template_600x600.vtu")

# path for input file 
Base_path = os.path.join(Code_path, "o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1")

# path to uncleaned vertices directory 
Vertices_Path = os.path.join(Base_path, "vertices_not_cleaned_NEW")


import numpy as np
import vtk
import matplotlib.pyplot as plt
import pandas as pd

from vtk_read_write import read_vtu, write_vtu
from vtk_convert import extract_data
from np_sorting_points import sort2d, sort2d_with_key
from vtk.util import numpy_support as VN
from vtk.util.numpy_support import numpy_to_vtk

import skfmm
from vtk_append_data import append_np_array 
from collections import defaultdict
from copy import deepcopy
import time


### used functions: 
def time_it(func):
    """
    Wrapper for functions for outputting the execution time of a function call. 
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

#------------- compute cell midpoints -----------
#!!!FRAME TIME IS HARDCODED
def all_my_midpoints(N_Cell):
    """
    Reads and stores the midpoints of cells at a specific time frame in a global variable.

    This function iterates over a series of cell data files, extracts the positions (`x0`, `x1`) 
    at time `t=20` for each cell, and stores these midpoints in the global variable `all_midpoints`.

    Args:
        base_file (str): The base directory containing the cell position data file.
        N_Cell (int): The total number of cells to process.

    Side Effects:
        - Creates or updates the global variable `all_midpoints`, 
          which is a 2D numpy array of shape `(N_Cell, 2)`. 
          Each row corresponds to the midpoint of a cell.

    Notes:
        - The function assumes the cell data CSV files have columns `time`, `x0`, and `x1`.
        - It specifically processes data for the time frame where `time == 20`.

    Example:
        base_path = "/path/to/data"
        N_cells = 10
        all_my_midpoints(base_path, N_cells)
        # After execution, `all_midpoints` contains the midpoints for all 10 cells.
    """
    global all_midpoints
    all_midpoints = np.zeros((N_Cell,2))
    
    for i in range(N_Cell):
        filename = f"neo_positions_p{i}.csv" 
        data_file = os.path.join(Base_path, "positions", filename)
        pos_phase = pd.read_csv(data_file)
        frame_time = pos_phase[pos_phase["time"]==20]
        x0 = frame_time["x0"].iloc[0]
        x1 = frame_time["x1"].iloc[0]
        all_midpoints[i,0] = x0
        all_midpoints[i,1] = x1
    return

def extract_to_smaller_file(path_phi, fine_grid, i, N_small_resolution):
    """        
    Saves all phi values of the domain

    Resamples a scalar field from a coarse source dataset onto a fine grid.

    Returns:
        np.ndarray: A 1d NumPy array containing the resampled scalar field values on the fine grid.
    """
    grid_fine = fine_grid.GetOutput()
    phi = read_vtu(path_phi).GetOutput()

    midpoint_selected = all_midpoints[i]
    print(f"selected midpoint = {midpoint_selected}")

    extracted_grid = extract_phi(phi, midpoint_selected, N_small_resolution)
    write_path_i = os.path.join(Output_path, f"fine_mesh_{i}_extracted.vtu")
    write_vtu(extracted_grid,write_path_i)

    # interpolate to fine grid 
    interpolator = vtk.vtkResampleWithDataSet()
    interpolator.SetInputData(grid_fine)
    interpolator.SetSourceData(extracted_grid)
    interpolator.Update()
    h=interpolator.GetOutput()

    write_path_i = os.path.join(Output_path, f"fine_mesh_{i}_interpolated.vtu")
    write_vtu(h,write_path_i)

    return VN.vtk_to_numpy(h.GetPointData().GetArray("phi"))

def extract_phi(phi, midpoint, N_small_resolution):
    """
    Extracts the scalar field `phi` from a given grid, considering periodic boundary conditions.
    The region containing midpoint must not be shifted! Somehow the interpolator cannot handle 
    this case later on. 

    Args:
        phi (vtkUnstructuredGrid): The VTK grid containing the scalar field.
        x_min, x_max (float): Bounds in the x-direction (can be outside the domain [0, Lx]).
        y_min, y_max (float): Bounds in the y-direction (can be outside the domain [0, Ly]).
        Lx, Ly (float): Domain size in the x and y directions (default is [0, 100]).

    Returns:
        np.ndarray: The scalar field values for the extracted subdomain considering periodicity.
    """
     
    # is the problem, that the section containing the midpoint gets shifted for i=5? 

    shift_index = round(N_small_resolution /20) 
    x_min = midpoint[0] - shift_index
    x_max = midpoint[0] + shift_index
    y_min = midpoint[1] - shift_index
    y_max = midpoint[1] + shift_index
        
    # Apply periodic boundary conditions to the bounds
    x_min_wrapped = x_min  % 100
    x_max_wrapped = x_max  % 100
    y_min_wrapped = y_min  % 100
    y_max_wrapped = y_max  % 100

    # Control x_interval
    x_intervals = []
    if x_min_wrapped < x_max_wrapped:
        # no wrapping needed 
        x_intervals.append([x_min_wrapped, x_max_wrapped])
    else:
        # Extract intervals from the left and right side of the domain
        x_intervals.append([x_min_wrapped, 101])
        x_intervals.append([-1, x_max_wrapped])
   
    # Control y_interval
    y_intervals = []
    if y_min_wrapped < y_max_wrapped:
        # no wrapping needed 
        y_intervals.append([y_min_wrapped, y_max_wrapped])
    else:
        # Extract intervals from the left and right side of the domain
        y_intervals.append([y_min_wrapped, 101])
        y_intervals.append([-1, y_max_wrapped])

    if len(x_intervals) == 1: 
        if len(y_intervals) == 1: 
            # everything is alright
            extracted_grid = extract_region(phi, x_min_wrapped, x_max_wrapped, y_min_wrapped, y_max_wrapped)
        elif len(y_intervals) == 2: 
            
            upper_grid = extract_region(phi, x_min_wrapped, x_max_wrapped, y_intervals[0][0], y_intervals[0][1]) 
            lower_grid = extract_region(phi, x_min_wrapped, x_max_wrapped, y_intervals[1][0], y_intervals[1][1]) 
            # concatenate vertically
            if is_midpoint_in_region(midpoint, x_min_wrapped, x_max_wrapped,  y_intervals[0][0], y_intervals[0][1]):
                # midpoint is in upper region
                # move lower grid over the upper grid 
                lower_grid = shift_grid_vtk(lower_grid, dy=100)
            else: 
                # midpoint is in lower region
                # move upper grid under the lower grid 
                upper_grid = shift_grid_vtk(upper_grid, dy=-100)
            extracted_grid = concatenate_grids(lower_grid, upper_grid) 


    elif len(x_intervals) == 2:
        if len(y_intervals) == 1: 
            # concatenate horizontally
            right_grid = extract_region(phi, x_intervals[0][0], x_intervals[0][1], y_min_wrapped, y_max_wrapped)
            left_grid = extract_region(phi, x_intervals[1][0], x_intervals[1][1], y_min_wrapped, y_max_wrapped)
            if is_midpoint_in_region(midpoint, x_intervals[0][0], x_intervals[0][1], y_min_wrapped, y_max_wrapped):
                # midpoint is in right region
                # move right grid next to the left grid 
                left_grid = shift_grid_vtk(left_grid, dx=100)
            else: 
                # midpoint is in left region
                # move right grid next to the left grid 
                right_grid = shift_grid_vtk(right_grid, dx=-100)
            extracted_grid = concatenate_grids(right_grid, left_grid) 

        elif len(y_intervals) == 2:
            # we have to concatenate 4 grids :-( 

            grid_upper_left = extract_region(phi, x_intervals[1][0], x_intervals[1][1], y_intervals[0][0], y_intervals[0][1]) 
            
            
            
            grid_lower_left = extract_region(phi, x_intervals[1][0], x_intervals[1][1], y_intervals[1][0], y_intervals[1][1])
            
            #write_path = os.path.join(Output_path, "lower_left_0.vtu")
            #write_vtu(grid_lower_left, write_path)
            


            grid_upper_right = extract_region(phi, x_intervals[0][0], x_intervals[0][1], y_intervals[0][0], y_intervals[0][1]) 
            grid_lower_right = extract_region(phi, x_intervals[0][0], x_intervals[0][1], y_intervals[1][0], y_intervals[1][1])
        	
            if is_midpoint_in_region(midpoint, x_intervals[0][0], x_intervals[0][1], y_intervals[0][0], y_intervals[0][1]):
                # midpoint is in upper right region 
                # so we have to move left to right and lower to high

                # left part: 
                # move upper grid under the lower grid 
                grid_lower_left = shift_grid_vtk(grid_lower_left, dy=100)
                left_extracted_grid = concatenate_grids(grid_upper_left, grid_lower_left)  
    
                # right part: 
                # move upper grid under the lower grid 
                grid_lower_right = shift_grid_vtk(grid_lower_right, dy=100)
                right_extracted_grid = concatenate_grids(grid_lower_right, grid_upper_right)
                
                # move left grid right to the left grid 
                left_extracted_grid = shift_grid_vtk(left_extracted_grid, dx=100)

                extracted_grid = concatenate_grids(right_extracted_grid, left_extracted_grid) 

            elif is_midpoint_in_region(midpoint, x_intervals[1][0], x_intervals[1][1], y_intervals[0][0], y_intervals[0][1]):
                # midpoint is in upper left region
                # so we have to move right to left and lower to high

                # left part: 
                # move lower grid over the upper grid 
                grid_lower_left = shift_grid_vtk(grid_lower_left, dy=100)
                left_extracted_grid = concatenate_grids(grid_upper_left, grid_lower_left)  
    
                # right part: 
                #  move lower grid over the upper grid 
                grid_lower_right = shift_grid_vtk(grid_lower_right, dy=100)
                right_extracted_grid = concatenate_grids(grid_lower_right, grid_upper_right)
                
                # move right grid left to the left grid 
                right_extracted_grid = shift_grid_vtk(right_extracted_grid, dx=-100)

                extracted_grid = concatenate_grids(right_extracted_grid, left_extracted_grid) 

            elif is_midpoint_in_region(midpoint, x_intervals[1][0], x_intervals[1][1], y_intervals[1][0], y_intervals[1][1]):
                # midpoint is in lower left region
                # so we have to move right to left and upper to low 

                # left part: 
                # move upper grid under the lower grid 
                grid_upper_left = shift_grid_vtk(grid_upper_left, dy=-100)

                left_extracted_grid = concatenate_grids(grid_upper_left, grid_lower_left)  
    
                # right part: 
                # move upper grid under the lower grid 
                grid_upper_right = shift_grid_vtk(grid_upper_right, dy=-100)
                right_extracted_grid = concatenate_grids(grid_lower_right, grid_upper_right)
                
                # move right grid left to the left grid 
                right_extracted_grid = shift_grid_vtk(right_extracted_grid, dx=-100)

                extracted_grid = concatenate_grids(right_extracted_grid, left_extracted_grid) 

            elif is_midpoint_in_region(midpoint, x_intervals[0][0], x_intervals[0][1], y_intervals[1][0], y_intervals[1][1]):
                # midpoint is in lower right region 
                # so we have to move left to right and upper to low

                # left part: 
                # move upper grid under the lower grid 
                grid_upper_left = shift_grid_vtk(grid_upper_left, dy=-100)
                left_extracted_grid = concatenate_grids(grid_upper_left, grid_lower_left)  
    
                # right part: 
                # move upper grid under the lower grid 
                grid_upper_right = shift_grid_vtk(grid_upper_right, dy=-100)
                right_extracted_grid = concatenate_grids(grid_lower_right, grid_upper_right)
                
                # move left grid right to the right grid 
                left_extracted_grid = shift_grid_vtk(left_extracted_grid, dx=100)

                extracted_grid = concatenate_grids(right_extracted_grid, left_extracted_grid) 
            else:
                raise ValueError("midpoint is not in the given area")

    # shift it so that midpoint -> 20,20 
    extracted_grid = shift_grid_vtk(extracted_grid, dx = shift_index - midpoint[0], dy = shift_index - midpoint[1])
    
    return extracted_grid

def is_midpoint_in_region(midpoint, x_min, x_max, y_min, y_max):
    return x_min <= midpoint[0] <= x_max and y_min <= midpoint[1] <= y_max

def shift_grid_vtk(grid, dx=0.0, dy=0.0, dz=0.0):
    """
    Shifts the coordinates of all points in a VTK grid using vtkTransform.

    Args:
        grid (vtkUnstructuredGrid): The input grid to shift.
        dx, dy, dz (float): The amount to shift in each direction.

    Returns:
        vtkUnstructuredGrid: The shifted grid.
    """
    # Create a transformation
    transform = vtk.vtkTransform()
    transform.Translate(dx, dy, dz)

    # Apply the transformation to the grid
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetInputData(grid)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    # Return the transformed grid
    return transform_filter.GetOutput()

def extract_region(phi, x_min, x_max, y_min, y_max):
    """
    Extracts a subregion of the grid `phi` based on the given bounds.

    Args:
        phi (vtkUnstructuredGrid): The input grid.
        x_min, x_max (float): Bounds in the x-direction. Must be in [0,100]. 
        y_min, y_max (float): Bounds in the y-direction. Must be in [0,100]. 

    Returns:
        vtkUnstructuredGrid: The extracted region.
    """
    # Define a box to extract a subregion
    box = vtk.vtkBox()
    box.SetBounds(x_min, x_max, y_min, y_max, 0, 0)

    # Set up the extractor
    extractor = vtk.vtkExtractGeometry()
    extractor.SetInputData(phi)
    extractor.SetImplicitFunction(box)
    extractor.ExtractInsideOn()
    extractor.Update()

    # Return the extracted region
    return extractor.GetOutput()

def concatenate_grids(grid_1, grid_2):
    """
    Concatenates two VTK grids horizontally (along the x-axis).

    Args:
        grid_1 (vtkUnstructuredGrid): The left grid to concatenate.
        grid_2 (vtkUnstructuredGrid): The right grid to concatenate.

    Returns:
        vtkUnstructuredGrid: A single grid containing the horizontally concatenated data.
    """
    append_filter = vtk.vtkAppendFilter()
    append_filter.AddInputData(grid_1)
    append_filter.AddInputData(grid_2)
    append_filter.Update()

    combined_grid = vtk.vtkUnstructuredGrid()
    combined_grid.DeepCopy(append_filter.GetOutput())

    return combined_grid 

def read_fine_grid(path_grid_file):
    """
    Reads the coordinates of a grid from a VTK file.

    Args:
        path_grid_file (str): Path to the VTK file containing the fine grid data.

    Returns:
        np.ndarray: The coordinates of the grid points as a 2D NumPy array.
    """
    coords_grid,_=extract_data(read_vtu(path_grid_file))
    return coords_grid

def resample_phi_on_fine_grid(filename,filename_grid):
    grid_fine=read_vtu(filename_grid)
    phi=read_vtu(filename)
    interpolator=vtk.vtkResampleWithDataSet()
    interpolator.SetInputData(grid_fine.GetOutput())
    interpolator.SetSourceData(phi.GetOutput())
    interpolator.Update()
    h=interpolator.GetOutput()
    return h

@time_it
def all_my_distances(N_small_resolution,N_Cell,value=0.2):
    """
    Computes and appends the unsigned distance field for multiple phases to a fine grid.

    This function processes multiple phase data files (`phase_p{i}_20.000.vtu`), resamples 
    the `phi` values onto a fine grid, calculates the unsigned distance for each phase using a 
    threshold value, and appends the results to the fine grid. The modified fine grid is then written 
    to an output file, and additional vertex information is computed.

    Args:
        base_file (str): Path to the directory containing the phase data files.
        N_small_resolution (int): The resolution of the new finer grid.
        N_Cell (int): The total number of phases/cells to process.
        file_grid (str): Path to the fine grid file.
        value (float, optional): The threshold value for calculating the unsigned distance field. 
                                 Default is 0.2.

    Returns:
        vtk.vtkPolyData: The modified fine grid with the appended unsigned distance fields.

    Example:
        fine_grid = all_my_distances(base_file, 100, 10, "fine_grid.vtu", 0.2)
    """
    
    # assume that just works maybe change N -> N +/- 1 
    coordinates_small_grid = read_fine_grid(Fine_grid_200x200_path)
    recalculate_indices(N_small_resolution,coordinates_small_grid)
    
    for i in range(N_Cell):
        
        print("resample loop ",i)
        small_grid_i = read_vtu(Fine_grid_200x200_path)
        phasefield_path = os.path.join(Base_path, "phasedata", f"phase_p{i}_20.000.vtu")
        phi_grid = extract_to_smaller_file(phasefield_path, small_grid_i, i, N_small_resolution)
        ud_i = calculate_unsigned_dist(40, phi_grid, value)
        small_grid_i = append_np_array(small_grid_i,ud_i,"ud_"+str(i))
        write_path_i = os.path.join(Output_path, f"fine_mesh_{i}_distance.vtu")
        write_vtu(small_grid_i, write_path_i)

    # Now all distances are computed and saved to all the small fine meshes. We want to transfer all to one big fine grid 

    # TODO: adapt all_my_vertices so it can deal with 100 small grid files
    all_my_vertices(N_Cell)
    
def compute_neighbor_indices(i):
    """
    Computes all cell indices that must be featured in the grid of cell i. i itself is included.  
    """
    res = [] 
    my_midpoint = all_midpoints[i] 
    for j in range(N_Cell):
        test_midpoint = all_midpoints[j] 
        if np.linalg.norm(test_midpoint - my_midpoint) <= 20.0:
            res.append(j)
    return res 

def discretize_to_big_file_ind(all_midpoints):
    res = np.zeros((N_Cell,2),dtype=int)
    for i in range(N_Cell):
        res[i] =round( all_midpoints[i] / dx) 
    return res 

def recalculate_indices(N,coords_grid):
    """
    Calculates and stores the indices mapping each point in a grid to its 
    corresponding position in a 2D index array. It also stores the x and y coordinates 
    of each grid point in separate 1D arrays.

    Args:
        N (int): The number of grid divisions (grid resolution).
        coords_grid (np.ndarray): A 2D NumPy array of shape (M, 2) containing the 
                                  coordinates of grid points, where M is the number 
                                  of points in the grid.

    Side Effects:
        - Updates global arrays:
            - `indices_phi`: A 2D array mapping grid positions to point indices.
            - `ind_phi_x`: A 1D array storing the x-coordinates of the grid points.
            - `ind_phi_y`: A 1D array storing the y-coordinates of the grid points.
            - `dx`: The grid spacing computed as the upper bound divided by `N`.

    Example:
        recalculate_indices(100, coords_grid)
        # Updates global variables with recalculated indices and coordinates.
    """
    global indices_phi,ind_phi_x,ind_phi_y,dx
    indices_phi=np.zeros((N+1,N+1),dtype=int)
    ind_phi_x=np.zeros((N+1)*(N+1),dtype=int)
    ind_phi_y=np.zeros((N+1)*(N+1),dtype=int)
    UpperBound=100.0
    dx=UpperBound/N
    for i in range(coords_grid.shape[0]):
        # i must be in range(( res = N+1 )^2)
        # in the new function, we have res = 41   
        x_coord=round(coords_grid[i,0]/dx)
        y_coord=round(coords_grid[i,1]/dx)
        indices_phi[x_coord,y_coord]=i
        ind_phi_x[i]=x_coord
        ind_phi_y[i]=y_coord

def calculate_unsigned_dist(N,phi,value):
    """
    Calculates the unsigned distance field based on a given scalar field and value.

    Args:
        N (int): The grid resolution.
        phi (np.ndarray): A 2D NumPy array representing the scalar field.
        value (float): The value used to compute the unsigned distance from the scalar field.

    Returns:
        np.ndarray: A 1D NumPy array containing the unsigned distance field.
    """

    phi_new = np.zeros((N+1,N+1))
    phi_new = phi[indices_phi]
    phi_new -= value
    unsigned_dist = skfmm.distance(phi_new,dx=np.array([dx,dx]),periodic=False)
    unsigned_dist_resorted = np.zeros((N+1)*(N+1))
    unsigned_dist_resorted = unsigned_dist[ind_phi_x,ind_phi_y]
    return unsigned_dist_resorted

def adjust_point(ref,test_h,grid_length=100):
    """
    Adjusts test point's coordinates to ensure they are within a 100-unit range of the reference point.
    """
    test=deepcopy(test_h)
    if (abs(test[0]-ref[0])>0.5*grid_length):
        if test[0] < ref[0]:
            test[0]+=grid_length
        else:
            test[0] -=grid_length
    if (abs(test[1]-ref[1])>0.5*grid_length):
        if test[1] < ref[1]:
            test[1]+= grid_length
        else:
            test[1] -= grid_length
    return test

def all_my_vertices(N_Cells,r=20.0):
    """
    Collects and saves the vertices of common boundaries between neighboring cells based on midpoints.

    This function iterates through multiple cells, computes potential neighbors based on the distance 
    between their midpoints, and identifies common boundaries between neighboring cells. For each 
    pair of neighboring cells, the function calculates the boundary, adjusts the coordinates if necessary, 
    and stores the vertices of the common boundary. These vertices are saved as `.npy` files for each cell.

    Args:
        fine_grid (vtk.vtkPolyData): The fine grid containing the scalar field data.
        N_Cells (int): The number of cells or phases to process.
        r (float, optional): The threshold distance for considering neighboring cells based on their midpoints.
                             Default is 20.0 units.

    Side Effects:
        - Saves `.npy` files for each cell's boundary vertices in the directory specified by `dir_vertices`.
    """

    # we need to be able to map the coords to the midpoint in the original grid 

    all_vertices_collected=defaultdict(list)

    for i in range(N_Cells):
        
        print(f"loop {i} in all_my_vertices")
        
        possible_neighs=[]
        my_midpoint_i=all_midpoints[i,:]
        for j in range(i+1, N_Cells):
            other_midpoint = all_midpoints[j,:]
            other_midpoint = adjust_point(my_midpoint_i,other_midpoint,100.0)
            if (np.linalg.norm(my_midpoint_i-other_midpoint)<r):
                possible_neighs.append(j)
     
        
        # now create operating grid 
        neighborhood_grid = read_vtu(Fine_grid_600x600_path).GetOutput()
        neighborhood_grid = shift_grid_vtk(
                                           neighborhood_grid, 
                                           dx = all_midpoints[i][0] - 30, 
                                           dy = all_midpoints[i][1] - 30
                                           )

        small_fine_grid_i_path = os.path.join(Output_path, f"fine_mesh_{i}_distance.vtu")
        fine_grid_i = read_vtu(small_fine_grid_i_path).GetOutput()
        # move it such that: midpoint_grid -> midpoint[i]; midpoint_grid = (10, 10)
        fine_grid_i = shift_grid_vtk(
                                     fine_grid_i, 
                                     dx=all_midpoints[i][0] - 10, 
                                     dy=all_midpoints[i][1] - 10
                                     )
        
        neighborhood_grid = append_small_grid_to_neighborhood_size(fine_grid_i, f"ud_{i}",neighborhood_grid)

        # append it here 
        neighborhood_grids_j = {}
        for j in possible_neighs:
            print(j)
            small_fine_grid_j_path = os.path.join(Output_path, f"fine_mesh_{j}_distance.vtu")
            fine_grid_j = read_vtu(small_fine_grid_j_path).GetOutput()
            # move it such that: midpoint_grid -> midpoint[i]; midpoint_grid = (10, 10)
            fine_grid_j = shift_grid_vtk(
                                         fine_grid_j, 
                                         dx=all_midpoints[j][0] - 10, 
                                         dy=all_midpoints[j][1] - 10
                                         )  
            neighborhood_grid = append_small_grid_to_neighborhood_size(fine_grid_j, f"ud_{j}",neighborhood_grid)
            neighborhood_grids_j[j] = fine_grid_j

        write_vtu(neighborhood_grid, os.path.join(Output_path, f"newNeighbors{i}.vtu"))
        # REMEMBER TO SEARCH VERTICES ONLY IN GRID_I \CAP GRID_J COORDINATES   
                
        print(f"now collecting all cells that are near midpoint[{i}]")

        for j in possible_neighs:
            # this excludes i 
            print(f"neighbor {j}")
            
            # operating subdomain = grid_i cap grid_j 
            x_min, x_max, y_min, y_max = cap_grids_bounds(fine_grid_i, neighborhood_grids_j[j])
            box = vtk.vtkBox()
            box.SetBounds([x_min, x_max, y_min, y_max, 0, 0])

            extractor = vtk.vtkExtractGeometry()
            extractor.SetInputData(neighborhood_grid)  # Input grid with arrays
            extractor.SetImplicitFunction(box)
            extractor.ExtractInsideOn()
            extractor.ExtractBoundaryCellsOn()
            extractor.Update()

            subdomain = extractor.GetOutput()
            write_vtu(subdomain, os.path.join(Output_path, f"subdomain_i{i}_j{j}.vtu"))
            # TODO: check whether coordinates are correct or need to be shifted to all_midpoints[i] 
            # compute diff between i and j 
            calculator = vtk.vtkArrayCalculator()
            calculator.SetInputData(subdomain)
            calculator.AddScalarVariable("i", f"ud_{i}_fixed")
            calculator.AddScalarVariable("j", f"ud_{j}_fixed")
            calculator.SetFunction("i-j")
            calculator.SetResultArrayName("diff")
            calculator.Update()
            calculator.GetOutput().GetPointData().SetActiveScalars("diff")

            # compute where diff is 0 (THESE POINTS ARE POSSIBLE VERTICES)
            contour = vtk.vtkContourFilter()
            contour.SetInputConnection(calculator.GetOutputPort())
            contour.SetValue(0,0)
            contour.Update()
            n_points = contour.GetOutput().GetNumberOfPoints()

            # collect the coordinates of these points
            coords = np.zeros((n_points,2))
            array_all = np.zeros((N_Cells,n_points))
            
            for k in range(n_points):
                coords[k,0], coords[k,1], _ = contour.GetOutput().GetPoint(k)

            for k in range(N_Cells):
                try:
                    array_all[k,:]=VN.vtk_to_numpy(contour.GetOutput().GetPointData().GetArray(f"ud_{k}_fixed"))
                except:
                    # TODO: check this sus value
                    array_all[k,:] = -2000
            
            array_all=array_all -array_all[i,:][None,:]
            indices=np.where(array_all.max(axis=0)<=0.0)[0]
            if len(indices)>0:
                midpoint_i=all_midpoints[i,:]
                midpoint_j=adjust_point(midpoint_i,all_midpoints[j,:],100.0)
            
                ref_vec_h=(midpoint_j-midpoint_i)/np.linalg.norm(midpoint_j-midpoint_i)
                refvec=np.zeros(2)
                refvec[0]=-ref_vec_h[1]
                refvec[1]=ref_vec_h[0]
                for k in indices:
                    coords[k,:]=adjust_point(midpoint_i,coords[k,:],100.0)
                
                coords_sorted,coords_keys=sort2d_with_key(coords[indices,:],midpoint_i,refvec)

                print(f"coords_sorted = {coords_sorted}, coords_keys = {coords_keys}")
                max_diff=0
                ind=-100
                for k in range(len(coords_keys)):
                    #TODO: check this line 
                    diff_curr=abs(np.arctan2(np.sin(coords_keys[k-1]-coords_keys[k]),np.cos(coords_keys[k-1]-coords_keys[k])))
                    if (max_diff<diff_curr):
                        max_diff=diff_curr
                        ind=k
                        print(ind, diff_curr)

                if ind != -100:
                    # adjust coordinates
                    if coords_sorted[ind,0] > 100.0:
                        coords_sorted[ind,0] -= 100.0
                    elif coords_sorted[ind,0] < 0.0:
                        coords_sorted[ind,0] += 100.0
                
                    if coords_sorted[ind,1] > 100.0:
                        coords_sorted[ind,1] -= 100.0
                    elif coords_sorted[ind,1] < 0.0:
                        coords_sorted[ind,1] += 100.0
                    
                    if coords_sorted[ind-1,0] > 100.0:
                        coords_sorted[ind-1,0] -= 100.0
                    elif coords_sorted[ind-1,0] < 0.0:
                        coords_sorted[ind-1,0] += 100.0
                    
                    if coords_sorted[ind-1,1] > 100.0:
                        coords_sorted[ind-1,1] -= 100.0
                    elif coords_sorted[ind-1,1] < 0.0:
                        coords_sorted[ind-1,1] += 100.0
                
                    all_vertices_collected[i].append(coords_sorted[ind,:])
                    all_vertices_collected[i].append(coords_sorted[ind-1,:])
                
                    all_vertices_collected[j].append(coords_sorted[ind,:])
                    all_vertices_collected[j].append(coords_sorted[ind-1,:])
                
            else:
                print("no common boundary")

        my_points_i=np.array(all_vertices_collected[i])
        np.save(f"{Vertices_Path}/phase_{i}",my_points_i)
        
        
def cap_grids_bounds(grid1, grid2):
    x_min1, x_max1, y_min1, y_max1, _, _ = grid1.GetBounds()
    x_min2, x_max2, y_min2, y_max2, _, _ = grid2.GetBounds()

    if x_min1 < x_min2: 
        x_min_res = x_min2
        x_max_res = x_max1 
    else: 
        x_min_res = x_min1
        x_max_res = x_max2 

    if y_min1 < y_min2: 
        y_min_res = y_min2
        y_max_res = y_max1 
    else: 
        y_min_res = y_min1
        y_max_res = y_max2 

    return x_min_res, x_max_res, y_min_res, y_max_res 

def append_small_grid_to_neighborhood_size(small_grid, array_name, neighborhood_grid): 

    """# Create a scalar array with default value -10
    default_array = vtk.vtkDoubleArray()
    default_array.SetName(array_name)
    default_array.SetNumberOfTuples(neighborhood_grid.GetNumberOfPoints())
    default_array.Fill(-10)  # Default value """

    # Add the default array to the neighborhood grid
    #    neighborhood_grid.GetPointData().AddArray(default_array)
    # neighborhood_grid.GetPointData().SetActiveScalars(array_name) 

    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetSourceData(small_grid)  
    probe_filter.SetInputData(neighborhood_grid)  
    probe_filter.Update()

    #write_vtu(probe_filter.GetOutput(), os.path.join(Output_path, "probefilter.vtu"))
    #print("probe filter info \n", probe_filter.GetOutput())


    calculator = vtk.vtkArrayCalculator()
    calculator.SetInputData(probe_filter.GetOutput())
    calculator.AddScalarVariable("ud", array_name, 0)
    calculator.AddScalarVariable("valids", "vtkValidPointMask", 0)
    calculator.SetFunction("ud+10*(valids-1)")
    calculator.SetResultArrayName(array_name+"_fixed")
    calculator.Update()

    interpolated_grid = calculator.GetOutput()

    # Transfer interpolated scalar values to the neighborhood grid
    interpolated_array = interpolated_grid.GetPointData().GetArray(array_name+"_fixed")
    if interpolated_array:
        neighborhood_grid.GetPointData().AddArray(interpolated_array)  # Add the interpolated array
        neighborhood_grid.GetPointData().SetActiveScalars(array_name+"_fixed")  # Activate the new scalar array
        return neighborhood_grid
    else:
        raise ValueError(f"Interpolation failed: '{array_name}' not found in the interpolated grid.")

def create_small_grid_template(grid_length, output_path, N_resolution):
    """
    Creates uniform triangular grids with specified resolution and saves them as `.vtu` files.

    Args:
        input_length_path (str): Path to the `.vtu` file used to determine the grid's length.
        output_path (str): Directory where the grids will be saved.
        N_Cells (int): Number of identical grids to generate.
        N_resolution (int): Resolution of the grids (N_resolution x N_resolution).

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure the output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Create the uniform grid points
        x_coords = np.linspace(0, grid_length, N_resolution)
        y_coords = np.linspace(0, grid_length, N_resolution)

        points = vtk.vtkPoints()
        for y in y_coords:
            for x in x_coords:
                points.InsertNextPoint(x, y, 0)

        # Define the triangular cells
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)

        for j in range(N_resolution - 1):
            for i in range(N_resolution - 1):
                pt1 = j * N_resolution + i
                pt2 = pt1 + 1
                pt3 = pt1 + N_resolution
                pt4 = pt3 + 1

                # Create two triangles for each square in the grid
                triangle1 = vtk.vtkTriangle()
                triangle1.GetPointIds().SetId(0, pt1)
                triangle1.GetPointIds().SetId(1, pt2)
                triangle1.GetPointIds().SetId(2, pt4)

                triangle2 = vtk.vtkTriangle()
                triangle2.GetPointIds().SetId(0, pt1)
                triangle2.GetPointIds().SetId(1, pt4)
                triangle2.GetPointIds().SetId(2, pt3)

                grid.InsertNextCell(triangle1.GetCellType(), triangle1.GetPointIds())
                grid.InsertNextCell(triangle2.GetCellType(), triangle2.GetPointIds())

        # Save the grid to `.vtu` files
        file_name = os.path.join(Code_path, f"small_fine_grid_template_{N_resolution}x{N_resolution}.vtu")
        print(f"grid bounds = {grid.bounds}")
        write_vtu(grid, file_name)
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


# THIS FILE CONTAINS MULTIPLE HARDCODED THINGS WHICH MIGHT BE BENEFICIAL TO REMOVE
# The periodicity of 100 is hardcoded
# Filenames for tests are hardcoded
# The time is hardcoded (20) in all_my_midpoints and also in resample on fine grid
# Path to vtk library is hardcoded
# calculateInnerContour is not used anymore in the main code in my opinion
# It has however a update now, that might it make worth to read: If you get any contour/cut/line from vtk it is not sorted. A month or so after I implemented this with this weird sorting algorithm (which only works nice for small convex cells - it should hold for all cells in the simulations, but this is kinda hard to prove and more a "I look at the data"-thing - I found a way to sort the points in the contour and I tested and implemented this in calculateInnerContour. You could try to use this, maybe with minor adjustmenst because of the periodicity, instead of the weird sorting algorithm
# A surely non optimal thing that I did is to implement everything on a global fine grid.
# Technical everything happens in a small surrounding of the cell. So a locally fine grid and/or smaller grid could be enough and would probably speed up everything


# GENERAL
# I also did not use i<j in my code and calculatedos all vertices twice and "cleaned" them later in another routine (because numerical the results are not exactly the same). This is also something that could be optimized
# All the vertices are stored in dir_vertices

# OTHER UGLY THING:
# I used global variables like all_midpoints, ind_phi_x, ind_phi_y


# WHAT DO YOU NEED TO EXECUTE HERE?
# all_my_midpoints
# all_my_distances

# WHAT DO YOU NEED FROM THE OTHER FILE
# clean_and_collect_my_vertices



# number of cells 

def asmain1():
    # old program process
    N_Cell = 5

    eps = 0.1
    filename = f"vertices_not_cleaned_eps_{eps}"
    global dir_vertices
    dir_vertices = os.path.join(Base_path, filename)
    if not os.path.exists(dir_vertices):
        os.mkdir(dir_vertices)

    # grid resolution
    N = 1000
    all_my_midpoints(N_Cell)
    print(all_midpoints)

    # c,d_a=calculateInnerContour(filename1)
    # print("c",c)
    # print("d_a",d_a)
    # plt.plot(d_a[:,0],d_a[:,1])
    # plt.show()
    # exit()

#### for creating the 100 small fine grids  
def build():
    N_fine_resolution = 600 
    grid_length = 60
    print(create_small_grid_template(grid_length, Output_path, N_fine_resolution))

global N_Cell  
N_Cell = 100
N_fine_resolution = 200 
eps = 0.1

# for i in range(N_Cell):
#     midpoint = all_midpoints[i]
#     if 30 < midpoint[0] < 70 and 30 < midpoint[1] < 70:
#         print(f"midpoint {i} = {midpoint}")

#all_my_midpoints(N_Cell)
# TODO: go on and check how all_my_distances works now 
#all_my_distances(N_fine_resolution, N_Cell)




