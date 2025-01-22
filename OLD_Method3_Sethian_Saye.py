import sys
import numpy as np
import vtk
import os
from pathlib import Path
home = Path.home()

#PATH TO Tims VTK Suite:
# C:\Users\voglt\OneDrive\Desktop\researchProject\code\vtk-suite
sys.path.append(os.path.join(home, "Onedrive", "Desktop", "researchProject", "code", "vtk-suite"))

Code_path = os.path.join(home, "Onedrive", "Desktop", "researchProject", "code")
Base_path = os.path.join(Code_path, "o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1")
Vertices_Path = os.path.join(Base_path, "vertices_not_cleaned_OLD_OUTPUT")

import matplotlib.pyplot as plt
import pandas as pd

from vtk_read_write import read_vtu, write_vtu
from vtk_convert import extract_data
from np_sorting_points import sort2d, sort2d_with_key
from vtk.util import numpy_support as VN
import skfmm
from vtk_append_data import append_np_array
from collections import defaultdict
from copy import deepcopy

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
        coords[i,0],coords[i,1],dummy_argument= contour.GetOutput().GetPoint(i)
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


#!!!FRAME TIME IS HARDCODED
def all_my_midpoints(base_file,N_Cell):
    """
    Computes and stores the midpoints of cells at a specific time frame in a global variable.

    This function iterates over a series of cell data files, extracts the positions (`x0`, `x1`) 
    at time `t=20` for each cell, and stores these midpoints in the global variable `all_midpoints`.

    Args:
        base_file (str): The base directory containing the cell position data files.
        N_Cell (int): The total number of cells to process.

    Side Effects:
        - Creates or updates the global variable `all_midpoints`, 
          which is a 2D numpy array of shape `(N_Cell, 2)`. 
          Each row corresponds to the midpoint of a cell.

    File Structure:
        - Assumes the cell data files are stored in a subdirectory called `positions` 
          under `base_file`, with filenames formatted as `neo_positions_p{i}.csv`.

    Returns:
        None

    Notes:
        - The function assumes the cell data CSV files have columns `time`, `x0`, and `x1`.
        - It specifically processes data for the time frame where `time == 20`.

    Example:
        base_path = "/path/to/data"
        total_cells = 10
        all_my_midpoints(base_path, total_cells)
        # After execution, `all_midpoints` contains the midpoints for all 10 cells.
    """
    print("Bla")
    global all_midpoints
    all_midpoints=np.zeros((N_Cell,2))
    
    for i in range(N_Cell):
        filename =f"neo_positions_p{i}.csv" 
        data_file = os.path.join(base_file, "positions", filename)
        pos_phase=pd.read_csv(data_file)
        frame_time=pos_phase[pos_phase["time"]==20]
        x0=frame_time["x0"].iloc[0]
        x1=frame_time["x1"].iloc[0]
        all_midpoints[i,0]=x0
        all_midpoints[i,1]=x1
    return

def interpolate_phi_on_fine_grid(filename,filename_grid):
    grid_fine=read_vtu(filename_grid)
    phi=read_vtu(filename)
    kernel=vtk.vtkGaussianKernel()
    kernel.SetNullValue(-1.0)
    interpolator=vtk.vtkPointInterpolator()
    interpolator.SetInputData(grid_fine.GetOutput())
    interpolator.SetSourceData(phi.GetOutput())
    interpolator.SetKernel(kernel)
    interpolator.Update()
    h=interpolator.GetOutput()
    print(h)

    write_vtu(h,r"C:\Users\voglt\OneDrive\Desktop\researchProject\test.vtu")
    return
    

def resample_phi_on_fine_grid(filename,filename_grid):
    grid_fine=read_vtu(filename_grid)
    phi=read_vtu(filename)
    print(f"type of phi = {type(phi)}")
    interpolator=vtk.vtkResampleWithDataSet()
    interpolator.SetInputData(grid_fine.GetOutput())
    interpolator.SetSourceData(phi.GetOutput())
    interpolator.Update()
    h=interpolator.GetOutput()
    #print(h)
    #write_vtu(h,"/Users/Lea.Happel/Documents/Software_IWR/pAticsProject/team-project-p-atics/fine_meshes/test_interpolation.vtu")
    return VN.vtk_to_numpy(h.GetPointData().GetArray("phi"))
    

def read_fine_grid(file_grid):
    coords_grid,dummy_argument=extract_data(read_vtu(file_grid))
    print("coords_grid ",coords_grid.shape)
    return coords_grid

def recalculate_indices(N,coords_grid):
    global indices_phi,ind_phi_x,ind_phi_y,dx
    indices_phi=np.zeros((N+1,N+1),dtype=int)
    ind_phi_x=np.zeros((N+1)*(N+1),dtype=int)
    ind_phi_y=np.zeros((N+1)*(N+1),dtype=int)
    UpperBound=100.0
    dx=UpperBound/N
    for i in range(coords_grid.shape[0]):
        x_coord=round(coords_grid[i,0]/dx)
        y_coord=round(coords_grid[i,1]/dx)
        indices_phi[x_coord,y_coord]=i
        ind_phi_x[i]=x_coord
        ind_phi_y[i]=y_coord
    print("i'm leaving")
    
def calculate_unsigned_dist(N,phi,value):
    phi_new=np.zeros((N+1,N+1))
    phi_new=phi[indices_phi]
    phi_new -=value
    unsigned_dist=skfmm.distance(phi_new,dx=np.array([dx,dx]),periodic=True)
    unsigned_dist_resorted=np.zeros((N+1)*(N+1))
    unsigned_dist_resorted=unsigned_dist[ind_phi_x,ind_phi_y]
    return unsigned_dist_resorted
    
def all_my_distances(base_file,N,N_Cell,file_grid,value=0.2):
    coords_grid=read_fine_grid(file_grid)
    fine_grid_new=read_vtu(file_grid)
    recalculate_indices(N,coords_grid)
    
    for i in range(N_Cell):
        print("resample loop ",i)
        filename = os.path.join(base_file, "phasedata", f"phase_p{i}_20.000.vtu")
        phi_grid=resample_phi_on_fine_grid(filename,file_grid)
        ud_i=calculate_unsigned_dist(N,phi_grid,value)
        fine_grid_new=append_np_array(fine_grid_new,ud_i,"ud_"+str(i))

    writePath = os.path.join(home, "Onedrive", "Desktop", "researchProject", "code", "output", "test_all_unsigned_dist.vtu")
    write_vtu(fine_grid_new, writePath)
    all_my_vertices(fine_grid_new,N_Cell)
    return fine_grid_new

def adjust_point(ref,test_h):
    test=deepcopy(test_h)
    if (abs(test[0]-ref[0])>50.0):
        if test[0] < ref[0]:
            test[0]+=100.0
        else:
            test[0] -=100.0
    if (abs(test[1]-ref[1])>50.0):
        if test[1] < ref[1]:
            test[1]+=100.0
        else:
            test[1] -=100.0
    return test

def all_my_vertices(fine_grid,N_Cells,r=20.0):
    all_vertices_collected=defaultdict(list)
    #Later on:Double loop
    for i in range(N_Cells):
        print("NCells ",i)
    #NOW: get all the indices for which the midpoints are close
        possible_neighs=[]
        my_midpoint_i=all_midpoints[i,:]
        for k in range(N_Cells):
            if i != k:
                other_midpoint=all_midpoints[k,:]
                other_midpoint=adjust_point(my_midpoint_i,other_midpoint)
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
                midpoint_j=adjust_point(midpoint_i,all_midpoints[j,:])
            
                ref_vec_h=(midpoint_j-midpoint_i)/np.linalg.norm(midpoint_j-midpoint_i)
                refvec=np.zeros(2)
                refvec[0]=-ref_vec_h[1]
                refvec[1]=ref_vec_h[0]
                for k in indices:
                    coords[k,:]=adjust_point(midpoint_i,coords[k,:])
                
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
        #np.save(dir_vertices+'/phase_'+str(i),my_points_i)
        np.save(f"{Vertices_Path}/phase_{i}",my_points_i)

        
    

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
base_file=os.path.join(home,"OneDrive", "Desktop", "researchProject", "code", "o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1")
# filename1='/Users/Lea.Happel/Downloads/o20230614_set3_In3Ca3aa0ar0D0v5Al0Ga3Init1/phasedata/phase_p45_20.000.vtu'
N_Cell=100
eps=0.1
filename = f"vertices_not_cleaned_eps_{eps}"
dir_vertices=os.path.join(base_file, filename)
file_grid=os.path.join(home,"OneDrive", "Desktop", "researchProject", "code", "grid_Harish_1000_1000.vtu")

N=1000
all_my_midpoints(base_file,N_Cell)
print(all_midpoints)
all_my_distances(base_file,N,N_Cell,file_grid,eps)

# c,d_a=calculateInnerContour(filename1)
# print("c",c)
# print("d_a",d_a)
# plt.plot(d_a[:,0],d_a[:,1])
# plt.show()
# exit()
