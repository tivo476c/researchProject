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
def clean_and_collect_my_vertices(base_vertices,N_Cell):
    clean_vertices=[]
    for i in range(N_Cell):
        clean_i=[]
        dirty_i=np.load(base_vertices+'/phase_'+str(i)+'.npy')
        clean_i.append(dirty_i[0,:])
        for j in range(1,dirty_i.shape[0]):
            append_j=True
            for k in clean_i:
                #shift dirty_i[j,:] to the same side as k
                dirty_vec_j=adjust_point(k,dirty_i[j,:])
                if np.linalg.norm(k-dirty_vec_j)<tol:
                    append_j=False
                    break
            if (append_j):
                clean_i.append(dirty_i[j,:])
        clean_vertices.append(np.array(clean_i))
        clean_array=np.array(clean_i)
        #print(clean_array.shape)
        #plt.scatter(clean_array[:,0],clean_array[:,1])
        #plt.show()
    return clean_vertices

### following functions are just for plotting and i must not touch them  ---------------------------------

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
    re, im = calculate_shapefct_components_with_midpoint(coords,midpoint,p)
    mag=np.sqrt(re*re+im*im)

    my_angle=np.arctan2(im,re)/p
    
    return mag,np.degrees(my_angle)

def plot_one_cell(coords,midpoint,magnitude,angle,p=3):
    #Assume that we have already manipulated coords
    coords_sorted=sort2d(coords,midpoint)
    x_star,y_star=get_pAtic_star(angle,p,6)
    corr=[-100,100,0]
    plot_coords=np.zeros((coords_sorted.shape[0]+1,2))
    plot_coords[:-1,:]=coords_sorted
    plot_coords[-1,:]=coords_sorted[0,:]
    for c in corr:
        for d in corr:
            plt.plot(x_star+midpoint[0]+c, y_star+midpoint[1]+d, linestyle='-', linewidth=2.0, color='r',alpha=magnitude)
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
    
# --------------------------------------------------------------------------------------------------------

#TODO: N_Cell=100
N_Cell=5
eps=0.1
tol=2*0.1*np.sqrt(2)
base_vertices = os.path.join(Base_path, 'vertices_not_cleaned_eps_'+str(eps))
all_my_midpoints(Base_path,N_Cell)
clean_and_collect_my_vertices(base_vertices,N_Cell)

#plot_m1(Base_path,base_vertices,"/Users/Lea.Happel/Documents/Software_IWR/pAticsProject/team-project-p-atics/Pictures_Data_Harish/m3_p_",N_Cell,[2,3,4,5,6])
#collect_and_post_process_m1(Base_path,base_vertices,Output_path,N_Cell,[2,3,4,5,6],'darkorange')
