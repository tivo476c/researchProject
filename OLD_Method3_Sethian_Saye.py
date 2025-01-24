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


[array([9.47696972, 4.3050189 ]), 
 array([11.0274868 , 95.30277109]), 
 array([99.70077881,  1.24382019]), 
 array([2.29516864, 5.84319687]), 
 array([ 6.70709467, 92.90211153]), 
 array([9.47696972, 4.3050189 ]), 
 array([11.0274868 , 95.30277109]), 
 array([13.47404385,  7.13799238]), 
 array([13.9730072 ,  7.13444281]), 
 array([20.4974556 ,  1.39758658]), array([17.08581352, 93.74235916]), array([37.42981339,  8.99440384]), array([32.88544846,  1.41540039]), array([26.03670311,  3.35168672]), array([26.45328903,  8.58025265]), array([31.9479599 , 13.91902637]), array([35.29614258, 14.0195303 ]), array([32.38293839, 14.37401104]), array([28.81101608,  0.85370034]), array([37.42981339,  8.99440384]), array([32.88544846,  1.41540039]), array([43.74132538,  7.0885973 ]), array([44.90924072,  1.42225075]), array([36.10701752, 97.14273143]), array([40.76062393, 97.00014067]), array([40.93592072, 96.69863296]), array([43.74132538,  7.0885973 ]), array([44.90924072,  1.42225075]), array([55.409832 ,  2.4341383]), array([51.70666885, 99.2717185 ]), array([48.17089081, 11.4793644 ]), array([53.87807083,  9.8853302 ]), array([55.409832 ,  2.4341383]), array([51.70666885, 99.2717185 ]), array([60.66196823, 99.89144897]), array([61.12622833, 91.98867798]), array([55.84790039, 88.33366394]), array([51.94357681, 91.01125336]), array([60.66196823, 99.89144897]), array([61.12622833, 91.98867798]), array([69.25571442, 99.98582458]), array([71.42513275, 92.95065308]), array([63.98878098,  2.19688416]), array([67.49571228, 89.25450134]), array([67.77477264, 89.30875397]), array([69.25571442, 99.98582458]), array([71.42513275, 92.95065308]), array([80.36569214,  1.52649689]), array([81.38449097, 97.11815643]), array([74.03157043,  4.0690918 ]), array([76.57663727, 92.08646393]), array([80.36569214,  1.52649689]), array([81.38449097, 97.11815643]), array([88.88713074,  6.65727949]), array([92.90723419, 99.8816074 ]), array([84.87597656,  6.85160398]), array([86.03205109, 95.04284763]), array([92.30421448, 97.44885206]), array([93.30924225, 99.78884503]), array([99.70077881,  1.24382019]), array([2.29516864, 5.84319687]), array([88.88713074,  6.65727949]), array([92.90723419, 99.8816074 ]), array([0.6621933, 9.6733675]), array([92.83049011, 10.77889442]), array([0.64414215, 9.37185955]), array([99.33451843,  1.17120039]), array([93.3283844,  0.1256281]), array([2.35050917, 5.86693573]), array([9.43084717, 4.39438915]), array([13.47404385,  7.13799238]), array([0.6621933, 9.6733675]), array([14.54808044, 11.28045464]), array([ 3.65290284, 13.87090111]), array([11.50672913, 14.47391605]), array([14.5254612 , 11.67459965]), array([13.9730072 ,  7.13444281]), array([20.4974556 ,  1.39758658]), array([26.03670311,  3.35168672]), array([26.45328903,  8.58025265]), array([14.54808044, 11.28045464]), array([13.56933689,  7.18613243]), array([20.42955017, 13.03869247]), array([26.33705902,  8.59239388]), array([31.9479599 , 13.91902637]), array([20.42955017, 13.03869247]), array([21.9444561 , 20.60923195]), array([25.89364433, 21.94159317]), array([32.34318542, 14.37807655]), array([35.29614258, 14.0195303 ]), array([37.42748642,  9.01382828]), array([43.66393661,  7.11170769]), array([48.17089081, 11.4793644 ]), array([46.17470932, 17.00127029]), array([38.34327316, 18.01405525]), array([45.70282364, 17.02728271]), array([48.20101166, 11.4793644 ]), array([53.87807083,  9.8853302 ]), array([46.17470932, 17.00127029]), array([58.15897751, 13.42309093]), array([50.20665741, 21.69092178]), array([57.57253647, 19.278862  ]), array([53.88174438,  9.87132359]), array([55.17059708,  2.51791692]), array([55.46116257,  2.4378891 ]), array([60.61231995, 99.99195862]), array([63.98878098,  2.19688416]), array([58.15897751, 13.42309093]), array([63.98243713, 10.3997097 ]), array([64.05617523,  2.22846985]), array([69.22800446,  0.2330246 ]), array([74.03157043,  4.0690918 ]), array([63.98243713, 10.3997097 ]), array([73.11107635, 10.86118221]), array([68.0953598 , 13.34424019]), array([74.06407166,  4.07151794]), array([80.33371735,  1.55653381]), array([84.87597656,  6.85160398]), array([73.11107635, 10.86118221]), array([81.33615875, 13.5560894 ]), array([77.08564758, 14.46061134]), array([77.4948349 , 14.15910339]), array([80.99298096, 13.59754372]), array([84.91887665,  6.89449978]), array([88.78662872,  6.73729658]), array([92.83049011, 10.77889442]), array([81.33615875, 13.5560894 ]), array([91.8118515 , 16.81262398]), array([85.93538666, 19.19802094]), array([92.86180878, 10.84395599]), array([0.64414215, 9.37185955]), array([ 3.65290284, 13.87090111]), array([91.8118515 , 16.81262398]), array([ 1.14263153, 20.61147499]), array([96.32113647, 21.21711159]), array([96.7826004 , 21.27656555]), array([ 3.67119122, 13.93867111]), array([11.50672913, 14.47391605]), array([ 1.14263153, 20.61147499]), array([12.46152115, 23.13353348]), array([11.51525974, 14.85058975]), array([ 5.7150836 , 25.32975769]), array([11.50686073, 14.47391605]), array([14.5254612 , 11.67459965]), array([14.57545471, 11.37463188]), array([20.37352943, 13.11817551]), array([21.9444561 , 20.60923195]), array([12.46152115, 23.13353348]), array([11.51525974, 14.85058975]), array([15.33944035, 25.21653748]), array([32.38293839, 14.37401104]), array([35.2662468, 14.0887661]), array([25.89364433, 21.94159317]), array([38.34327316, 18.01405525]), array([36.78793335, 23.17971039]), array([27.25371742, 26.49629211]), array([30.07788277, 27.48064995]), array([38.36614227, 18.00860596]), array([45.70282364, 17.02728271]), array([46.16732788, 17.03595734]), array([50.20665741, 21.69092178]), array([36.78793335, 23.17971039]), array([48.75401688, 25.90882492]), array([41.31585312, 27.65298462]), array([50.21928024, 21.69774437]), array([57.57253647, 19.278862  ]), array([48.75401688, 25.90882492]), array([62.01376724, 23.51601601]), array([52.12587357, 30.01003075]), array([58.50925064, 30.16225052]), array([57.59457779, 19.21697617]), array([58.18864059, 13.44971561]), array([63.95868301, 10.45837307]), array([68.0953598 , 13.34424019]), array([62.01376724, 23.51601601]), array([66.73468018, 22.2394371 ]), array([68.12715149, 13.23433018]), array([73.04878998, 10.86048794]), array([77.08564758, 14.46061134]), array([66.73468018, 22.2394371 ]), array([75.50019836, 22.49689865]), array([70.14950562, 25.00946236]), array([70.31695557, 24.60745239]), array([77.4948349 , 14.15910339]), array([80.99298096, 13.59754372]), array([81.33387756, 13.56987953]), array([85.93538666, 19.19802094]), array([75.50019836, 22.49689865]), array([77.05898285, 14.51812077]), array([84.50299835, 26.11254311]), array([81.69477081, 27.46835899]), array([85.98270416, 19.20513916]), array([91.78556824, 16.86017609]), array([96.32113647, 21.21711159]), array([84.50299835, 26.11254311]), array([92.71822357, 29.05025101]), array([96.74774933, 21.31155777]), array([96.7826004 , 21.27656555]), array([ 1.12454987, 20.69389534]), array([ 5.7150836 , 25.32975769]), array([92.71822357, 29.05025101]), array([ 4.31656647, 29.75367546]), array([94.49210358, 32.13634872]), array([ 4.28717804, 30.07410812]), array([ 5.81352329, 25.34206772]), array([12.46907616, 23.21920395]), array([15.33944035, 25.21653748]), array([ 4.31656647, 29.75367546]), array([15.96452713, 32.11349106]), array([ 4.34770775, 30.1081295 ]), array([ 7.04425812, 33.78525162]), array([12.44693947, 34.90409088]), array([16.26603508, 32.11948776]), array([21.9740448 , 20.64702988]), array([25.98366928, 21.80478477]), array([15.41139984, 25.21099472]), array([27.25371742, 26.49629211]), array([15.96452713, 32.11349106]), array([20.92925644, 33.5412941 ]), array([16.30930519, 32.10573196]), array([27.25740623, 26.59679604]), array([30.07788277, 27.48064995]), array([20.92925644, 33.5412941 ]), array([32.07239914, 35.70420456]), array([23.59454155, 39.58815765]), array([27.39036369, 40.38235474]), array([30.16061211, 27.4835453 ]), array([36.76264572, 23.23130798]), array([41.31585312, 27.65298462]), array([32.07239914, 35.70420456]), array([40.32907867, 33.14064789]), array([35.23237991, 36.63167953]), array([41.34213638, 27.45198059]), array([48.66353226, 25.91884232]), array([52.12587357, 30.01003075]), array([40.32907867, 33.14064789]), array([47.32907104, 38.82542801]), array([48.9371109 , 38.38370895]), array([58.50925064, 30.16225052]), array([62.01230621, 23.70140076]), array([66.73355865, 22.322258  ]), array([70.14950562, 25.00946236]), array([70.22164917, 30.50708961]), array([60.38698196, 33.63724518]), array([66.54288483, 34.16500473]), array([70.31695557, 24.60745239]), array([75.45094299, 22.53542137]), array([81.69477081, 27.46835899]), array([70.22164917, 30.50708961]), array([70.17751312, 25.03581429]), array([76.94985199, 35.13828278]), array([80.95458984, 33.41438675]), array([76.94985199, 35.13828278]), array([80.95458984, 33.41438675]), array([85.49268341, 37.2511673 ]), array([75.94815826, 42.63267136]), array([85.9309082 , 45.04151154]), array([75.98352814, 43.07004929]), array([82.55072784, 47.25256729]), array([81.74582672, 27.47013283]), array([84.4430542 , 26.21304512]), array([92.69346619, 29.11214447]), array([94.49210358, 32.13634872]), array([80.96731567, 33.39844513]), array([85.49268341, 37.2511673 ]), array([93.21583557, 35.07715225]), array([94.53842926, 32.14663315]), array([ 4.28717804, 30.07410812]), array([ 7.04425812, 33.78525162]), array([93.21583557, 35.07715225]), array([ 2.07790756, 41.01984024]), array([ 6.60052013, 33.93684006]), array([97.06140018, 40.59659195]), array([ 7.05675554, 33.78525162]), array([12.44693947, 34.90409088]), array([ 2.07790756, 41.01984024]), array([ 6.60052013, 33.93684006]), array([13.00854969, 42.51221848]), array([ 4.76734304, 46.34182358]), array([ 8.38543415, 46.90494919]), array([12.5107069 , 34.95454407]), array([16.26603508, 32.11948776]), array([20.86532021, 33.5412941 ]), array([23.59454155, 39.58815765]), array([13.00854969, 42.51221848]), array([18.38725662, 45.05356979]), array([27.39036369, 40.38235474]), array([32.08026505, 35.74578857]), array([35.23237991, 36.63167953]), array([38.77894974, 45.44258499]), array([29.44916153, 47.901474  ]), array([35.60969543, 49.00088501]), array([35.29885864, 36.65314484]), array([40.25750732, 33.16664124]), array([47.32907104, 38.82542801]), array([38.77894974, 45.44258499]), array([43.51227951, 45.83597565]), array([47.39958572, 38.87666702]), array([48.9371109 , 38.38370895]), array([43.51227951, 45.83597565]), array([54.36836624, 43.16407394]), array([47.20230484, 52.70899963]), array([52.05680847, 52.23456573]), array([52.2305603 , 30.21522141]), array([58.4805603 , 30.18148994]), array([49.02324295, 38.38977432]), array([60.38698196, 33.63724518]), array([54.36836624, 43.16407394]), array([57.03911209, 41.96276474]), array([60.64744186, 33.69669724]), array([66.54288483, 34.16500473]), array([57.03911209, 41.96276474]), array([60.26707458, 33.81186676]), array([68.18240356, 42.53675461]), array([61.40998459, 45.9736824 ]), array([66.60995483, 34.17618942]), array([70.15222168, 30.71448708]), array([76.91855621, 35.20568848]), array([75.94815826, 42.63267136]), array([68.18240356, 42.53675461]), array([72.01052856, 44.9908905 ]), array([75.93965912, 43.04294968]), array([85.9309082 , 45.04151154]), array([85.57949066, 37.3028183 ]), array([93.01483154, 35.08951187]), array([93.43469191, 35.1234436 ]), array([97.06140018, 40.59659195]), array([92.38038635, 47.08433533]), array([97.15328455, 40.69207001]), array([ 2.03329587, 41.04587555]), array([ 4.76734304, 46.34182358]), array([92.38038635, 47.08433533]), array([99.58643341, 52.60124207]), array([93.74337769, 51.17052078]), array([ 4.79337645, 46.34006882]), array([ 8.38543415, 46.90494919]), array([99.58643341, 52.60124207]), array([11.32210922, 54.9046669 ]), array([ 3.04452944, 59.58512878]), array([ 8.44539452, 46.87651062]), array([12.99952221, 42.58641434]), array([18.38725662, 45.05356979]), array([11.32210922, 54.9046669 ]), array([19.63336563, 51.05184555]), array([14.63331509, 55.94270706]), array([23.6717701 , 39.64575958]), array([27.28986168, 40.37318039]), array([18.48775864, 45.08771133]), array([29.44916153, 47.901474  ]), array([19.63336563, 51.05184555]), array([24.92935181, 52.9185524 ]), array([19.9134388 , 51.09906769]), array([29.47904396, 47.92907333]), array([35.60969543, 49.00088501]), array([24.92935181, 52.9185524 ]), array([37.12334061, 55.76136398]), array([35.7665596 , 49.26036072]), array([26.16356659, 57.46990967]), array([31.70572281, 59.53972626]), array([35.77766418, 49.20800781]), array([38.76784515, 45.48941422]), array([43.4644165 , 45.86412811]), array([47.20230484, 52.70899963]), array([37.12334061, 55.76136398]), array([42.65582657, 58.36673355]), array([43.01014709, 58.38036728]), array([52.05680847, 52.23456573]), array([54.37910843, 43.26176453]), array([57.03409576, 41.97464371]), array([61.40998459, 45.9736824 ]), array([61.77300644, 52.8950119 ]), array([55.64235306, 56.90934753]), array([58.42498016, 56.93774414]), array([61.4487381 , 46.00799942]), array([68.28291321, 42.54110718]), array([72.01052856, 44.9908905 ]), array([61.77300644, 52.8950119 ]), array([71.54238892, 53.49646378]), array([75.98352814, 43.07004929]), array([82.55072784, 47.25256729]), array([72.11103058, 45.05857086]), array([71.54238892, 53.49646378]), array([81.97843933, 53.5919342 ]), array([73.03371429, 55.14143753]), array([82.57810974, 47.4163475 ]), array([85.93536377, 45.08218384]), array([92.28398895, 47.06896591]), array([93.74337769, 51.17052078]), array([81.97843933, 53.5919342 ]), array([89.83835602, 56.88716888]), array([85.85515594, 57.37069702]), array([93.76706696, 51.16944122]), array([99.57373047, 52.50074005]), array([ 3.04452944, 59.58512878]), array([89.83835602, 56.88716888]), array([ 2.57054901, 61.28771973]), array([95.81048584, 63.27136993]), array([ 3.21197701, 59.38412476]), array([11.33969402, 54.96201324]), array([14.63331509, 55.94270706]), array([ 2.57054901, 61.28771973]), array([ 3.0819397 , 59.63970947]), array([16.14622116, 62.96275711]), array([ 7.69076729, 66.73358917]), array([14.73381805, 55.99716568]), array([19.884655  , 51.17742157]), array([24.9236927, 52.9937439]), array([26.16356659, 57.46990967]), array([16.14622116, 62.96275711]), array([20.46916771, 65.46160889]), array([26.21882057, 57.55018997]), array([31.70572281, 59.53972626]), array([20.46916771, 65.46160889]), array([32.68863678, 66.64617157]), array([20.61083221, 66.01485443]), array([27.78172493, 70.15317535]), array([32.99814224, 66.71837616]), array([31.73134995, 59.47995758]), array([37.13292313, 55.8211937 ]), array([42.65582657, 58.36673355]), array([32.68863678, 66.64617157]), array([43.62873459, 62.21559906]), array([42.9991951 , 58.39650345]), array([32.97806931, 66.70019531]), array([38.14171219, 68.17845154]), array([47.2326889 , 52.74246597]), array([51.97861481, 52.2287941 ]), array([43.01014709, 58.38036728]), array([55.64235306, 56.90934753]), array([43.62873459, 62.21559906]), array([49.76280975, 64.22181702]), array([55.68188477, 56.90821838]), array([58.42498016, 56.93774414]), array([49.76280975, 64.22181702]), array([60.9549675 , 61.92036819]), array([50.60649109, 68.8388443 ]), array([55.41569138, 71.85848236]), array([57.22017288, 70.85650635]), array([58.55692673, 56.90536118]), array([61.89339828, 53.04959106]), array([71.43871307, 53.49329758]), array([73.03371429, 55.14143753]), array([60.9549675 , 61.92036819]), array([72.35552979, 59.03656769]), array([65.68751526, 63.05204391]), array([73.10174561, 55.16519547]), array([81.87793732, 53.59620285]), array([85.85515594, 57.37069702]), array([72.35552979, 59.03656769]), array([83.41983032, 63.16031265]), array([76.98892212, 63.96557999]), array([85.92369843, 57.3946228 ]), array([89.74906158, 56.89837646]), array([95.81048584, 63.27136993]), array([83.41983032, 63.16031265]), array([94.35227966, 67.79972076]), array([86.61022949, 68.44158173]), array([95.8571701, 63.3180542]), array([ 2.56215668, 61.34825134]), array([ 7.69076729, 66.73358917]), array([94.35227966, 67.79972076]), array([ 6.69791317, 70.75205231]), array([97.85369158, 72.26683807]), array([ 7.77806187, 66.76864624]), array([16.14076042, 63.07104111]), array([20.41117096, 65.23706818]), array([20.5443821 , 65.51234436]), array([20.61083221, 66.01485443]), array([ 6.69791317, 70.75205231]), array([16.8009243 , 73.51351166]), array([ 9.62346554, 74.18430328]), array([20.62833405, 66.04686737]), array([27.78172493, 70.15317535]), array([16.8009243 , 73.51351166]), array([28.06762695, 76.10494995]), array([19.18226433, 77.85054016]), array([24.77335548, 79.0358963 ]), array([27.83989143, 70.1354599 ]), array([32.99814224, 66.71837616]), array([38.14171219, 68.17845154]), array([28.06762695, 76.10494995]), array([39.65005875, 73.47425842]), array([34.28458786, 78.70899963]), array([38.16576004, 68.16929626]), array([43.64753723, 62.29429245]), array([49.71486282, 64.24041748]), array([50.60649109, 68.8388443 ]), array([39.65005875, 73.47425842]), array([43.6229248 , 74.78955078]), array([50.68751144, 68.83885956]), array([55.41569138, 71.85848236]), array([43.6229248 , 74.78955078]), array([46.08262634, 81.09030914]), array([52.88641739, 80.65791321]), array([55.09529114, 71.91201019]), array([57.22017288, 70.85650635]), array([60.82394028, 61.99034882]), array([65.68751526, 63.05204391]), array([68.63453674, 71.97748566]), array([64.84599304, 75.46269226]), array([65.7003479 , 63.25304794]), array([72.35090637, 59.12296295]), array([76.98892212, 63.96557999]), array([68.63453674, 71.97748566]), array([74.14649963, 71.84781647]), array([77.01702118, 63.99367523]), array([83.39863586, 63.18531036]), array([86.61022949, 68.44158173]), array([74.14649963, 71.84781647]), array([84.08037567, 73.3896637 ]), array([76.78500366, 74.5303421 ]), array([84.03401184, 73.0819931 ]), array([86.6524353 , 68.46091461]), array([94.30201721, 67.83856964]), array([97.85369158, 72.26683807]), array([84.08037567, 73.3896637 ]), array([95.94441223, 76.82221985]), array([84.00801849, 73.10333252]), array([87.45005798, 77.57409668]), array([97.91194177, 72.28089905]), array([ 6.62551975, 70.7432251 ]), array([ 9.62346554, 74.18430328]), array([95.94441223, 76.82221985]), array([ 6.78123093, 80.64637756]), array([98.41568339, 80.56634521]), array([ 9.66524601, 74.22003174]), array([16.72927284, 73.50963593]), array([19.18226433, 77.85054016]), array([ 6.78123093, 80.64637756]), array([14.90943432, 84.60921478]), array([ 9.08028793, 84.60596466]), array([19.23267555, 77.9009552 ]), array([24.77335548, 79.0358963 ]), array([14.90943432, 84.60921478]), array([18.92953491, 77.9172821 ]), array([26.47329903, 87.22220612]), array([18.88784218, 89.86813354]), array([24.64014435, 89.8917923 ]), array([26.7816658, 87.4101944]), array([24.79117393, 79.03734589]), array([28.06892776, 76.18444061]), array([34.28458786, 78.70899963]), array([26.47329903, 87.22220612]), array([35.85299683, 84.7325592 ]), array([26.90544128, 87.44612885]), array([32.6562767 , 88.45115662]), array([34.37647247, 78.72123718]), array([39.72963715, 73.57476044]), array([43.56574249, 74.78955078]), array([46.08262634, 81.09030914]), array([35.85299683, 84.7325592 ]), array([42.81158447, 85.82175446]), array([55.84790039, 88.33366394]), array([51.94357681, 91.01125336]), array([46.16144943, 81.16912842]), array([52.88641739, 80.65791321]), array([42.81158447, 85.82175446]), array([56.27679062, 83.77960968]), array([42.82752991, 86.19167328]), array([44.37801361, 89.53981018]), array([55.42674255, 71.85391998]), array([57.11967087, 70.90750885]), array([52.97882843, 80.64982605]), array([55.09529114, 71.91201019]), array([57.44629288, 70.90952301]), array([64.84599304, 75.46269226]), array([56.27679062, 83.77960968]), array([64.84494781, 79.99446869]), array([64.55762482, 80.09065247]), array([64.89551544, 75.41171265]), array([68.63520813, 72.0150528 ]), array([74.115448  , 71.87846375]), array([76.78500366, 74.5303421 ]), array([64.84494781, 79.99446869]), array([75.31554413, 80.82821655]), array([68.460289 , 83.0760498]), array([74.92701721, 80.84169769]), array([76.78534698, 74.53068542]), array([84.03401184, 73.0819931 ]), array([87.45005798, 77.57409668]), array([75.31554413, 80.82821655]), array([84.56338501, 83.48638916]), array([79.09368134, 84.64985657]), array([87.46943665, 77.79447937]), array([95.92321777, 76.8456192 ]), array([98.41568339, 80.56634521]), array([84.56338501, 83.48638916]), array([95.60954285, 85.7470932 ]), array([87.4936142 , 86.95311737]), array([98.60195196, 80.36534119]), array([ 6.68974447, 80.65539551]), array([ 9.08028793, 84.60596466]), array([95.60954285, 85.7470932 ]), array([ 5.6154561 , 89.82415009]), array([98.18284667, 89.72822571]), array([11.02722454, 95.30250978]), array([ 6.70709467, 92.90211153]), array([17.08581352, 93.74235916]), array([ 9.16221333, 84.67124176]), array([14.84281445, 84.62319946]), array([18.88784218, 89.86813354]), array([ 5.6154561 , 89.82415009]), array([20.48387146,  1.28349972]), array([17.16263008, 93.74115276]), array([28.81101608,  0.85370034]), array([26.04404449,  3.2657609 ]), array([18.93561172, 89.88951111]), array([24.64014435, 89.8917923 ]), array([32.71311188,  1.3848809 ]), array([28.81258011,  0.85370034]), array([36.10701752, 97.14273143]), array([24.71761513, 89.86875916]), array([26.7816658, 87.4101944]), array([32.6562767 , 88.45115662]), array([32.70336533, 88.74956512]), array([40.76062393, 97.00014067]), array([36.26593399, 97.12187743]), array([32.79089737, 88.65216064]), array([35.86528015, 84.82536316]), array([42.74053192, 85.83706665]), array([42.82752991, 86.19167328]), array([44.37801361, 89.53981018]), array([35.9495163 , 97.05227661]), array([40.92152405, 96.70570374]), array([44.90915298,  1.36539364]), array([40.93592072, 96.69863296]), array([51.682724  , 99.24777472]), array([51.90828323, 91.01194   ]), array([44.47851562, 89.40111542]), array([61.11595535, 91.95175171]), array([55.86116028, 88.33164978]), array([67.49571228, 89.25450134]), array([56.30396271, 83.8462677 ]), array([64.55762482, 80.09065247]), array([64.84755707, 80.10362244]), array([68.460289 , 83.0760498]), array([71.39286041, 92.92741394]), array([67.77477264, 89.30875397]), array([76.57663727, 92.08646393]), array([68.55904388, 82.91768646]), array([74.92701721, 80.84169769]), array([75.40078735, 80.85645294]), array([79.09368134, 84.64985657]), array([68.55491638, 83.31391144]), array([81.34020233, 97.03392029]), array([76.67713928, 92.10493469]), array([86.03205109, 95.04284763]), array([79.09407806, 84.6502533 ]), array([84.53964233, 83.66365051]), array([87.4936142 , 86.95311737]), array([92.30421448, 97.44885206]), array([86.07305908, 95.06901169]), array([87.61083984, 86.96221924]), array([95.57472992, 85.77128601]), array([98.18284667, 89.72822571]), array([ 6.61677694, 92.8763237 ]), array([99.69328997,  1.15450406]), array([93.30924225, 99.78884503]), array([92.32684326, 97.61904597]), array([99.33451843,  1.17120039]), array([93.3283844,  0.1256281]), array([98.3803941 , 89.72476959]), array([ 5.59477758, 89.84279633]), array([98.02029419, 89.76070404])]