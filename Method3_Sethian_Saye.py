"""
PREREQUISITS: the following directories and files must be saved in the executing system: 
* vtk-suite
* phase field input file "o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1"
* fine grid file "grid_Harish_1000_1000.vtu"
* .vtu file for saving unsigned distances "test_all_unsigned_dist.vtu"
* directory that holds all fine grids for all cells 

THE PATHS TO THESE FILES MUST BE SET MANUALLY IN THE NEXT LINES 
"""

from pathlib import Path
import os
import sys

home = Path.home()

# path to VTK suite:
VTK_path = os.path.join(home, "Onedrive", "Desktop", "researchProject", "code", "vtk-suite")
sys.path.append(VTK_path)

# path for input file 
Base_path = os.path.join(home, "OneDrive", "Desktop", "researchProject", "code", "o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1")

# path to grid 
Grid_path = os.path.join(home,"OneDrive", "Desktop", "researchProject", "code", "grid_Harish_1000_1000.vtu")

# path for saving output/test_all_unsigned_dist.vtu
Write_path = os.path.join(home, "Onedrive", "Desktop", "researchProject", "code", "output", "test_all_unsigned_dist.vtu")

# path to fine grid directory 
Fine_grids_path = os.path.join(home, "Onedrive", "Desktop", "researchProject", "code", "all_new_fine_grids")

# path to input example file 
Example_grid_path = os.path.join(Base_path, "phasedata", "phase_p45_20.000.vtu")

'/Users/Lea.Happel/Downloads/o20230614_set3_In3Ca3aa0ar0D0v5Al0Ga3Init1/phasedata/phase_p45_20.000.vtu'


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


### not used function: 

def calculateInnerContour(filename,value=0.2):
    """
    Calculates and extracts the inner contour of a 2D scalar field from a VTK unstructured grid file.

    The function reads a `.vtu` file, extracts a specific isosurface (contour) based on the provided scalar value, 
    and processes the contour lines into structured numpy arrays. It uses VTK for reading and processing 
    the unstructured grid data.

    Args:
        - filename (str): Path to the `.vtu` file containing the unstructured grid data.
        - value (float, optional): Scalar value for the contour extraction. Defaults to 0.2.

    Returns:
        - store_p (list of numpy.ndarray): A list of arrays where each array represents 
          the coordinates of a connected contour segment in 2D space.
        - coords (numpy.ndarray): A 2D array of shape `(n_points, 2)` containing 
          the sorted coordinates of the contour points in 2D space.
    """
    reader=vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    data=reader.GetOutput()
    
        #Get contour for plotting
    attr=reader.GetOutput().GetAttributes(0).GetArray("phi")
    reader.GetOutput().GetPointData().SetScalars(attr)

    # create contour filter
    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(reader.GetOutputPort())
    contour.SetValue(0,value)
    contour.Update()
    print(contour.GetOutput())
    stripper=vtk.vtkStripper()
    stripper.SetInputData(contour.GetOutput())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    print("stripper")
    print(stripper.GetOutput())
    #contour=stripper
    n_points = contour.GetOutput().GetNumberOfPoints()
    coords = np.zeros((n_points,2))
    for i in range(n_points):
        coords[i,0],coords[i,1],_= contour.GetOutput().GetPoint(i)
    lines=stripper.GetOutput().GetLines()
    points=stripper.GetOutput().GetPoints()
    lines.InitTraversal()
    idList=vtk.vtkIdList()
    store_p=[]
    all_indices_p=[]
    while lines.GetNextCell(idList):
        p=[]
        for i in range(0,idList.GetNumberOfIds()):
            print(i)
            p.append(points.GetPoint(idList.GetId(i)))
            all_indices_p.append(idList.GetId(i))
        p_arr=np.array(p)
        store_p.append(p_arr)
    #midpoint=np.mean(coords,axis=0)
    #print(midpoint)
    #coords=sort2d(coords,midpoint)
    
    return store_p,coords[np.array(all_indices_p),:]

def interpolate_phi_on_fine_grid(name_coarse_grid, name_fine_grid):
    """
    Interpolates a the cell model for a given coarser grid on a finer grid. 
    """
    grid_fine=read_vtu(name_fine_grid)
    phi=read_vtu(name_coarse_grid)
    kernel=vtk.vtkGaussianKernel()
    kernel.SetNullValue(-1.0)
    interpolator=vtk.vtkPointInterpolator()
    interpolator.SetInputData(grid_fine.GetOutput())
    interpolator.SetSourceData(phi.GetOutput())
    interpolator.SetKernel(kernel)
    interpolator.Update()
    h=interpolator.GetOutput()
    print(h)
    # set write path as you want 
    write_vtu(h,r"test.vtu")
    return


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
def all_my_midpoints(base_file,N_Cell):
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
        data_file = os.path.join(base_file, "positions", filename)
        pos_phase = pd.read_csv(data_file)
        frame_time = pos_phase[pos_phase["time"]==20]
        x0 = frame_time["x0"].iloc[0]
        x1 = frame_time["x1"].iloc[0]
        all_midpoints[i,0] = x0
        all_midpoints[i,1] = x1
    return

def resample_phi_on_fine_grid(name_coarse_grid, name_fine_grid):
    """        
    Just Input here: subdomain of interest as name_coarse_grid -> how to cut out subdomain?

    Resamples a scalar field from a coarse source dataset onto a fine grid.

    Returns:
        np.ndarray: A 1d NumPy array containing the resampled scalar field values on the fine grid.
    """
    grid_fine=read_vtu(name_fine_grid)
    phi=read_vtu(name_coarse_grid)
    interpolator=vtk.vtkResampleWithDataSet()
    interpolator.SetInputData(grid_fine.GetOutput())
    interpolator.SetSourceData(phi.GetOutput())
    interpolator.Update()
    h=interpolator.GetOutput()
    res = VN.vtk_to_numpy(h.GetPointData().GetArray("phi"))
    print(f"Shape of res in resample_phi_on_fine_grid: {res.shape}")
    return res 
    
def read_fine_grid(path_grid_file):
    """
    Reads the coordinates of a grid from a VTK file.

    Args:
        path_grid_file (str): Path to the VTK file containing the fine grid data.

    Returns:
        np.ndarray: The coordinates of the grid points as a 2D NumPy array.
    """
    coords_grid,_=extract_data(read_vtu(path_grid_file))
    print("coords_grid.shape = ", coords_grid.shape)
    return coords_grid

@time_it
def all_my_distances(base_file,N,N_Cell,file_grids,value=0.2):
    """
    Computes and appends the unsigned distance field for multiple phases to a fine grid.

    This function processes multiple phase data files (`phase_p{i}_20.000.vtu`), resamples 
    the `phi` values onto a fine grid, calculates the unsigned distance for each phase using a 
    threshold value, and appends the results to the fine grid. The modified fine grid is then written 
    to an output file, and additional vertex information is computed.

    Args:
        base_file (str): Path to the directory containing the phase data files.
        N (int): The resolution of the new finer grid.
        N_Cell (int): The total number of phases/cells to process.
        file_grid (str): Path to the fine grid file.
        value (float, optional): The threshold value for calculating the unsigned distance field. 
                                 Default is 0.2.

    Returns:
        vtk.vtkPolyData: The modified fine grid with the appended unsigned distance fields.

    Example:
        fine_grid = all_my_distances(base_file, 100, 10, "fine_grid.vtu", 0.2)
    """
    coords_grid = read_fine_grid(file_grids)
    fine_grid_new = read_vtu(file_grids)
    recalculate_indices(N,coords_grid)
    
    """
    THE NEW PLAN IS TO: 
        * call resample_phi_on_fine_grid with a 40x40 small fine grid as second argument 
        * we need to do either ... or ...:
            - use coordinates from big grid on small grid somehow st. the small grid doesnt start at (0,0) but at midpoint - (20,20)
            - change the coords obtained from big grid to: coords -> coords - midpoint + (20,20)
        * in that small grid should all cells j != i be saved that have norm(midpoint[i] - midpoint[j]) <= 20.0 (which is also old value for considering neighbors)
    """ 
    
    for i in range(N_Cell):
        print("resample loop ",i)
        filename = os.path.join(base_file, "phasedata", f"phase_p{i}_20.000.vtu")
        phi_grid=resample_phi_on_fine_grid(filename,file_grids)
        ud_i=calculate_unsigned_dist(N,phi_grid,value)
        fine_grid_new=append_np_array(fine_grid_new,ud_i,"ud_"+str(i))

    write_vtu(fine_grid_new, Write_path)

    all_my_vertices(fine_grid_new,N_Cell)

    return fine_grid_new

def discretize_to_big_file_ind(all_midpoints):
    res = np.zeros((N_Cell,2),dtype=int)
    for i in range(N_Cell):
        res[i] =round( all_midpoints[i] / dx) 
    return res 

def is_point_in_subdomain(point, subdomain_center)->bool:
    return  (subdomain_center[0]-200 <= point[0] <= subdomain_center[0]+200 and \
             subdomain_center[1]-200 <= point[1] <= subdomain_center[1]+200      )

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
    print(f"in recalculate_indices we have: coords_grid.shape[0] = {coords_grid.shape[0]}")
    for i in range(coords_grid.shape[0]):
        # i must be in range(( res = N+1 )^2)
        # in the new function, we have res = 41   
        x_coord=round(coords_grid[i,0]/dx)
        y_coord=round(coords_grid[i,1]/dx)
        indices_phi[x_coord,y_coord]=i
        ind_phi_x[i]=x_coord
        ind_phi_y[i]=y_coord
        if 0 <= i <= 1000: 
            print(f"coordinates of {i}th point = ({x_coord}, {y_coord})")

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
    unsigned_dist = skfmm.distance(phi_new,dx=np.array([dx,dx]),periodic=True)
    unsigned_dist_resorted = np.zeros((N+1)*(N+1))
    unsigned_dist_resorted = unsigned_dist[ind_phi_x,ind_phi_y]
    return unsigned_dist_resorted

def adjust_point(ref,test_h,grid_length):
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

def all_my_vertices(fine_grid,N_Cells,r=20.0):
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
    all_vertices_collected=defaultdict(list)
    #Later on:Double loop
    for i in range(N_Cells):
        print("NCells ",i)
    #NOW: get all the indices for which the midpoints are close
        possible_neighs=[]
        my_midpoint_i=all_midpoints[i,:]
        # TODO: do k>i here!!!
        for k in range(i+1, N_Cells):
        #for k in range(N_Cells):
            print(f"i = {i}, k = {k}")
            
            other_midpoint=all_midpoints[k,:]
            other_midpoint=adjust_point(my_midpoint_i,other_midpoint,100.0)
            if (np.linalg.norm(my_midpoint_i-other_midpoint)<r):
                possible_neighs.append(k)
            
        for j in possible_neighs:
            print("current j is ",j)
            calculator = vtk.vtkArrayCalculator()
            calculator.SetInputData(fine_grid.GetOutput())
            calculator.AddScalarVariable("i","ud_"+str(i), 0)
            calculator.AddScalarVariable("j","ud_"+str(j), 0)
            calculator.SetFunction("i-j")
            calculator.SetResultArrayName("diff")
            calculator.Update()
            calculator.GetOutput().GetPointData().SetActiveScalars("diff")
            contour = vtk.vtkContourFilter()
            contour.SetInputConnection(calculator.GetOutputPort())
            contour.SetValue(0,0)
            contour.Update()
            n_points = contour.GetOutput().GetNumberOfPoints()
            coords = np.zeros((n_points,2))
            array_all=np.zeros((N_Cells,n_points))
            for k in range(n_points):
                coords[k,0],coords[k,1],dummy_argument= contour.GetOutput().GetPoint(k)
            for k in range(N_Cells):
                array_all[k,:]=VN.vtk_to_numpy(contour.GetOutput().GetPointData().GetArray("ud_"+str(k)))
            array_all=array_all -array_all[i,:][None,:]
            indices=np.where(array_all.max(axis=0)<=0.0)[0]
            print("i ", i ,"j ",j)
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
    
                max_diff=0
                ind=-100
                for k in range(len(coords_keys)):
                    diff_curr=abs(np.arctan2(np.sin(coords_keys[k-1]-coords_keys[k]),np.cos(coords_keys[k-1]-coords_keys[k])))
                    if (max_diff<diff_curr):
                        #print(ind, diff_curr)
                        max_diff=diff_curr
                        ind=k
                        print(ind, diff_curr)
            
                #adjust coordinates
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
            
            else:
                print("no common boundary")
        my_points_i=np.array(all_vertices_collected[i])
        np.save(dir_vertices+'/phase_'+str(i),my_points_i)

def create_and_save_uniform_triangular_grids(grid_length, output_path, N_Cells, N_resolution):
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
        for i in range(N_Cells):
            file_name = os.path.join(output_path, f"fine_mesh_{i}.vtu")

            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(file_name)
            writer.SetInputData(grid)
            writer.Write()

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def read_vtu_length(path):
    """
    Reads a `.vtu` file and calculates the length of the grid.
    
    Args:
        path (str): Path to the `.vtu` file.

    Returns:
        float: Length of the grid.
    """
    try:
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(path)
        reader.Update()
        grid = reader.GetOutput()

        # Extract bounding box
        bounds = grid.GetBounds()  # [xmin, xmax, ymin, ymax, zmin, zmax]
        length_x = bounds[1] - bounds[0]
        length_y = bounds[3] - bounds[2]
        length = max(length_x, length_y)

        return length

    except Exception as e:
        print(f"Error reading VTU length: {e}")
        return None

#---------------small functions-----------------
def recalculate_indices_small(N, coords_grid, i):
    """
    Calculates and stores the indices mapping each point in a grid to its 
    corresponding position in a 2D index array. It also stores the x and y coordinates 
    of each grid point in separate 1D arrays.

    Args:
        N (int): The number of grid divisions (grid resolution).
        coords_grid (np.ndarray): This array holds all coordinates from the big grid. 
                                  A 2D NumPy array of shape (M, 2) containing the 
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
    
    UpperBound=100.0
    dx=UpperBound/N
    small_grid_size = 40
    float_center = all_midpoints[i] # thats the float 2d mid point of considered cell i
    center = [round(float_center[0]/dx), round(float_center[1]/dx)] # thats its coordinates in old big file

    indSmallFile = 0 
    for small_x in range(center[0]-20, center[0]+21):
        for small_y in range(center[1]-20, center[1]+21):
            # {(small_x, small_y)} is the set of all points of our subdomain of interest 
            # TODO: check if this works 
            # assumption: the grid is stored row major like 
            # k = 1001 * small_x + 
           
            # make sure they are in (0,1000)^2 
            if small_x < 0:
                small_x += 1000
            elif small_x > 1000:
                small_x -= 1000
            if small_y < 0:
                small_y += 1000
            elif small_y > 1000:
                small_y -= 1000
            
            indBigFile = 1001*small_x + small_y 

            x_coord=round(coords_grid[indBigFile,0]/dx)
            y_coord=round(coords_grid[indBigFile,1]/dx)
            indices_phi[i][x_coord, y_coord] = indSmallFile
            ind_phi_x[i][indSmallFile] = x_coord
            ind_phi_y[i][indSmallFile] = y_coord

            indSmallFile += 1
        # WE HAVE TO SET:  indices_phi[i], ind_phi_x[i], ind_phi_y[i]
    #print("i'm leaving")

def all_my_distances_small(base_file,N,N_Cell,fine_grid_new,value=0.2):
    """ TODO: 
    Takes the big input file coords_grid of length 100x100, iterates i = range(NCell) and 
    creates, with each iteration representing one cell, a grid of size 40x40 that is a subdomain
    of coords_grid and stores all information of the subdomain with area 40x40 and midpoint = all_midpoints[i] 
    in a new grid file fine_mesh_{i}.vtu. 

    Computes and appends the unsigned distance field for multiple phases to a fine grid.

    This function processes multiple phase data files (`phase_p{i}_20.000.vtu`), resamples 
    the `phi` values onto a fine grid, calculates the unsigned distance for each phase using a 
    threshold value, and appends the results to the fine grid. The modified fine grid is then written 
    to an output file, and additional vertex information is computed.
    
    Side effects:
        creates global variables:
            - `indices_phi`: A vector of length NCell with each element being a 
            2D array mapping grid positions to point indices.
            - `ind_phi_x`: A vector of length NCell with each element being a 
            1D array storing the x-coordinates of the grid points.
            - `ind_phi_y`: A vector of length NCell with each element being a
            1D array storing the y-coordinates of the grid points.
            - `dx`: The grid spacing computed as the upper bound divided by `N`.

    Args:
        base_file (str): Path to the directory containing the phase data files.
        N (int): The resolution of the new finer grid.
        N_Cell (int): The total number of phases/cells to process.
        file_grid (str): Path to the fine grid file.
        value (float, optional): The threshold value for calculating the unsigned distance field. 
                                 Default is 0.2.

    Returns:
        vtk.vtkPolyData: The modified fine grid with the appended unsigned distance fields.

    Example:
        fine_grid = all_my_distances(base_file, 100, 10, "fine_grid.vtu", 0.2)
    """
    

    global indices_phi,ind_phi_x,ind_phi_y,dx
    indices_phi = [np.zeros((40, 40), dtype=int) for _ in range(N_Cell)]    # for each phase (N_Cell phases)
                                                 # elemens are: indicis_phi[cell][] = 
    ind_phi_x = [np.zeros(40 * 40, dtype=int) for _ in range(N_Cell)]      # flattened x-coordinates
    ind_phi_y = [np.zeros(40 * 40, dtype=int) for _ in range(N_Cell)]      # flattened y-coordinates
    # dx is global, initialized before processing
    dx = 100.0 / 1000  # Assuming N=1000
    # maybe we need path to harish1000x1000 grif file instead 
    coords_grid = read_fine_grid(base_file)
    int_all_midpoints = discretize_to_big_file_ind(all_midpoints)

    for ind_big_file in range(coords_grid.shape[0]): 

        x_coord=round(coords_grid[ind_big_file,0]/dx)
        y_coord=round(coords_grid[ind_big_file,1]/dx)
    
        for i in range(N_Cell):
            if is_point_in_subdomain([x_coord, y_coord], int_all_midpoints[i]):
                append_point_insubdomain(i)



        recalculate_indices_small(N,coords_grid,i)
        print("resample loop ",i)
        filename = os.path.join(base_file, "phasedata", f"phase_p{i}_20.000.vtu")
        phi_grid=resample_phi_on_fine_grid(filename,file_grids)
        ud_i=calculate_unsigned_dist(N,phi_grid,value)
        fine_grid_new=append_np_array(fine_grid_new,ud_i,"ud_"+str(i))

    write_vtu(fine_grid_new, Write_path)
    all_my_vertices(fine_grid_new,N_Cell)
    return fine_grid_new

 
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
# TODO: reset N_Cell = 100 

def main():
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
    all_my_midpoints(Base_path,N_Cell)
    print(all_midpoints)
    #  TODO: go on 
    all_my_distances(Base_path,N,N_Cell,Grid_path,eps)

    # c,d_a=calculateInnerContour(filename1)
    # print("c",c)
    # print("d_a",d_a)
    # plt.plot(d_a[:,0],d_a[:,1])
    # plt.show()
    # exit()

#### playground 
def build():
    old_vtu_length = read_vtu_length(Example_grid_path) # its 100
    N_fine_resolution = 400 
    grid_length = 0.4 * old_vtu_length
    eps = 0.1

    create_and_save_uniform_triangular_grids(grid_length, Fine_grids_path, N_Cell, N_fine_resolution)
    all_my_distances(Base_path,N_fine_resolution,N_Cell,Fine_grids_path,eps)

    recalculate_indices_small
    # TODO: next: go inside of all_my_distances(Base_path,N,N_Cell,Grid_path,eps) and adjust it so that it works with the fine meshes that are created 

global N_Cell  
N_Cell = 5 
main()