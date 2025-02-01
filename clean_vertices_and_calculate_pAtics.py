"""
PREREQUISITS: the following directories and files must be saved in the executing system: 
* vtk-suite
* phase field input file "o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1"
* output directory
* Method3_Sethian_Saye.py must be in the same directory as this file 

THE PATHS TO THESE FILES MUST BE SET MANUALLY IN THE NEXT LINES 
"""

import sys
import os
from pathlib import Path
import vtk

home = Path.home()
# path to VTK suite:
VTK_path = os.path.join(home, "Onedrive", "Desktop", "researchProject", "code", "vtk-suite")
sys.path.append(VTK_path)

# path for input file 
Base_path = os.path.join(home, "OneDrive", "Desktop", "researchProject", "code", "o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1")
# Base_path='/Users/Lea.Happel/Downloads/o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1/'

# path for plot_m1 to save the plot to 
Output_path = os.path.join(home, "OneDrive", "Desktop", "researchProject", "code", "output") 
# Pic_path = "/Users/Lea.Happel/Documents/Software_IWR/pAticsProject/team-project-p-atics/Pictures_Data_Harish/m3_p_"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from np_sorting_points import sort2d
# from pfold_orientation_methods import calculate_shapefct_components_with_midpoint
# from plotting_methods import get_pAtic_star

from Method3_Sethian_Saye import all_my_midpoints, adjust_point, time_it

@time_it
def clean_and_collect_my_vertices(base_vertices,N_Cell, sampleTime = 20.0):
    tol=2*0.1*np.sqrt(2)
    # tol=0.4
    clean_vertices=[]
    for i in range(N_Cell):
        #print(f"i in clean and collect my vertices = {i}")
        clean_i=[] 
        dirty_i=np.load(os.path.join(base_vertices, f"dirty_vertices_time_{sampleTime}", f"phase_{i}.npy"))
        #print(f"dirty_i = {dirty_i}")
        #print(dirty_i[0,:])
        clean_i.append(dirty_i[0,:])
        for j in range(1,dirty_i.shape[0]):
            append_j=True 
            for k in clean_i:
                #shift dirty_i[j,:] to the same side as k
                dirty_vec_j=adjust_point(k,dirty_i[j,:],100)
                if np.linalg.norm(k-dirty_vec_j)<tol:
                    append_j=False
                    break

            if (append_j):
                clean_i.append(dirty_i[j,:])
        clean_vertices.append(np.array(clean_i))

        
        clean_array=np.array(clean_i)
        #print(clean_array.shape)
    
    
    collected_vertices = collect_to_1array(clean_vertices)
    filtered_vertices = filter_double_points(collected_vertices)

    #print(f"clean_vertices = \n {clean_vertices}")
    #print(f"dims_clean_vertices = {np.ndim(clean_vertices)}")
    #print(f"shape_clean_vertices = {np.shape(clean_vertices)}")
    print(f"size collected_vertices = {np.size(collected_vertices)}")
    print(f"len collected_vertices = {len(collected_vertices)}")
    
    print(f"size filtered_vertices = {np.size(filtered_vertices)}")
    print(f"len filtered_vertices = {len(filtered_vertices)}")


    color = "orange" 
    plt.figure(figsize=(8, 8))
    # SCATTER POINTS with old clean_vetices  -----------------------------------------------------------------------
    # for _, cell_points in enumerate(clean_vertices):
    #     cell_points = np.array(cell_points)  # Ensure it's a NumPy array
    #     x_coords = cell_points[:, 0]
    #     y_coords = cell_points[:, 1]
    #     plt.scatter(x_coords, y_coords, label=False, color=color)

    # SCATTER POINTS with new filtered_vertices  -----------------------------------------------------------------------
    for x, y in filtered_vertices:
        plt.scatter(x, y, label=False, color=color)
    
    # CELL CONTOURS ------------------------------------------------------------------------
    for i in range(N_Cell):
        _, arr = calculateInnerContour(os.path.join(Base_path, "phasedata", f"phase_p{i}_20.000.vtu"))
        grouped_arr = group_vertices(arr)
        for array in grouped_arr:
            plt.plot(array[:,0],array[:,1], label=False, color=color)


    plt.xlim(0, 100)  
    plt.ylim(0, 100)  
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio

    return clean_vertices


def collect_to_1array(clean_vertices):

    #print(f"np.count_nonzero(clean_vertices) = {np.count_nonzero(clean_vertices)}")
    joinedArr = []
    # for arr in clean_vertices:
    for arr in clean_vertices: 
        # print(f"arr = {arr}")
        for point in arr: 
            joinedArr.append(point)

    # print(f"joinedArr = \n {joinedArr}")
    return joinedArr


def filter_double_points(clean_vertices):
    res = []
    for point in clean_vertices:
        append = True
        for collected_point in res:
            # print(f"point = {point}")
            # print(f"collected_point = {collected_point}")
            if np.linalg.norm(point-collected_point) < 0.2: 
                append = False 
        if append:
            res.append(point)

    return res 



### following functions are just for plotting and i must not touch them  ---------------------------------

def group_vertices(input_array, max_distance=40):
    """
    Groups vertices into clusters such that all vertices in a cluster are within `max_distance` of each other.

    Args:
        argument (np.ndarray): Input array of shape (n, 2), where n is the number of points.
        max_distance (float): Maximum allowable distance between any two points in the same cluster.

    Returns:
        list of np.ndarray: List of arrays, where each array contains points belonging to one cluster.
    """
    n_points = input_array.shape[0]
    if n_points == 0:
        return []  # Return an empty list if there are no points

    remaining_indices = set(range(n_points))  # Indices of points not yet assigned to a cluster
    clusters = []

    while remaining_indices:
        # Start a new cluster
        cluster = []
        seed_index = remaining_indices.pop()  # Take an arbitrary point as the seed for the cluster
        cluster.append(input_array[seed_index])

        # Check distances for the rest of the points
        to_check = [seed_index]
        while to_check:
            current_index = to_check.pop()
            current_point = input_array[current_index]

            # Find all points within max_distance of the current point
            for idx in list(remaining_indices):  # Convert to list to safely iterate and modify
                if np.linalg.norm(input_array[idx] - current_point) <= max_distance:
                    cluster.append(input_array[idx])
                    to_check.append(idx)
                    remaining_indices.remove(idx)

        # Add the cluster as a NumPy array to the result
        clusters.append(np.array(cluster))

    return clusters


def m1_for_one_set(coords,midpoint,p=3):
    n_points=coords.shape[0]
    for i in range(n_points):
        if (abs(coords[i,0]-midpoint[0])>50.0):
            if coords[i,0] < midpoint[0]:
                coords[i,0]+=100.0
            else:
                coords[i,0] -=100.0
        if (abs(coords[i,1]-midpoint[1])>50.0):
            if coords[i,1] < midpoint[1]:
                coords[i,1]+=100.0
            else:
                coords[i,1] -=100.0
    # re, im = calculate_shapefct_components_with_midpoint(coords,midpoint,p)
    mag=np.sqrt(re*re+im*im)

    my_angle=np.arctan2(im,re)/p
    
    return mag,np.degrees(my_angle)

def plot_one_cell(coords,midpoint,magnitude,angle,p=3):
    #Assume that we have already manipulated coords
    coords_sorted=sort2d(coords,midpoint)
    #x_star,y_star=get_pAtic_star(angle,p,6)
    corr=[-100,100,0]
    plot_coords=np.zeros((coords_sorted.shape[0]+1,2))
    plot_coords[:-1,:]=coords_sorted
    plot_coords[-1,:]=coords_sorted[0,:]
    for c in corr:
        for d in corr:
            #plt.plot(x_star+midpoint[0]+c, y_star+midpoint[1]+d, linestyle='-', linewidth=2.0, color='r',alpha=magnitude)
            plt.plot(plot_coords[:,0]+c,plot_coords[:,1]+d,color='k')
    

def plot_m1(dirname,base_vertices,pic_filename,NumberOfCells,p_vec):
    all_my_midpoints(dirname,NumberOfCells)
    clean_vertices=clean_and_collect_my_vertices(base_vertices,NumberOfCells)
    for p in p_vec:
        print("M3 (VII) current p ",p)
        mag_vec=np.zeros(NumberOfCells)
        ang_vec=np.zeros(NumberOfCells)
        for phase in range(NumberOfCells):
            mag,ang=m1_for_one_set(clean_vertices[phase],all_midpoints[phase,:],p)
            plot_one_cell(clean_vertices[phase],all_midpoints[phase,:],mag,ang,p)
            mag_vec[phase]=mag
            ang_vec[phase]=ang
        print("min and max ang_vec ",np.min(ang_vec)," ",np.max(ang_vec))
        print("mean mag ", np.mean(mag_vec))
        print("median mag ", np.median(mag_vec))
        print("std mag ", np.std(mag_vec))
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        #plt.savefig("Pictures_Data_Harish/m3_"+str(p)+pic_filename+".png",bbox_inches='tight',dpi=1000,transparent=True)
        #plt.show()

def collect_and_post_process_m1(dirname,base_vertices,pic_filename,NumberOfCells,p_vec,my_color):
    all_my_midpoints(dirname,NumberOfCells)
    clean_vertices=clean_and_collect_my_vertices(base_vertices,NumberOfCells)
    for p in p_vec:
        print("M3 (VII) current p ",p)
        mag_vec=np.zeros(NumberOfCells)
        ang_vec=np.zeros(NumberOfCells)
        for phase in range(NumberOfCells):
            mag,ang=m1_for_one_set(clean_vertices[phase],all_midpoints[phase,:],p)
            #plot_one_cell(clean_vertices[phase],all_midpoints[phase,:],mag,ang,p)
            mag_vec[phase]=mag
            ang_vec[phase]=ang
        print("min and max ang_vec ",np.min(ang_vec)," ",np.max(ang_vec))
        
        N=len(mag_vec)
        nr_bin=int(N/10)
        bins_array=np.linspace(-np.pi/p,np.pi/p,nr_bin,endpoint=False)
        print("before ",bins_array)
        bins_array += 0.5*(bins_array[1]-bins_array[0])
        print("after ",bins_array)
    
        percentage=np.zeros((nr_bin))
        ang_rad=np.radians(ang_vec)
        for i in range(nr_bin-1):
            percentage[i]=np.sum((ang_rad>=bins_array[i]) &(ang_rad<bins_array[i+1]))/N
        percentage[-1]=(np.sum(ang_rad>=bins_array[-1])+np.sum(ang_rad<bins_array[0]))/N
        print("max percentage ",np.max(percentage))
        print("sum percentage ",np.sum(percentage))
        ax=plt.subplot(projection='polar')
        ax.set_theta_zero_location("E")
        ax.set_xticks([-np.pi/p,-0.5*np.pi/p,0.0,0.5*np.pi/p,np.pi/p])
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        shift=0.5*(bins_array[1]-bins_array[0])
        ax.set_ylim((0.0,0.22))
        ax.set_xlim((-np.pi/p,np.pi/p))
        print("bins_array+shift ",bins_array+shift)
        ax.bar(bins_array+shift,percentage,width=2*shift,bottom=0.0,alpha=1.0,color=my_color)
        ax.bar(bins_array[0]-shift,percentage[-1],width=2*shift,bottom=0.0,alpha=1.0,color=my_color)
        plt.tight_layout()
        plt.savefig(pic_filename+str(p)+'.png',transparent=True,dpi=600)
        plt.savefig(pic_filename+str(p)+'.svg',transparent=True,dpi=600)
        plt.show()

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
    #print(contour.GetOutput())
    stripper=vtk.vtkStripper()
    stripper.SetInputData(contour.GetOutput())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    #print("stripper")
    #print(stripper.GetOutput())
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
            #print(i)
            p.append(points.GetPoint(idList.GetId(i)))
            all_indices_p.append(idList.GetId(i))
        p_arr=np.array(p)
        store_p.append(p_arr)
    #midpoint=np.mean(coords,axis=0)
    #print(midpoint)
    #coords=sort2d(coords,midpoint)
    #print(f"all_indices_p = {all_indices_p}")

    return store_p,coords[np.array(all_indices_p),:]


# --------------------------------------------------------------------------------------------------------

@time_it 
def startCleaning(dirty_vertices_dir, output_clean_path=""):
    N_Cell=100
    global tol 
    tol=2*0.1*np.sqrt(2)
    clean_and_collect_my_vertices(dirty_vertices_dir, N_Cell) 
    plt.show()

startCleaning(os.path.join(Base_path, "vertices_not_cleaned_Final_1_extract_phi2"))

