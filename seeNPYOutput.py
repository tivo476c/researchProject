import numpy as np
import matplotlib.pyplot as plt
import os 
from pathlib import Path

from clean_vertices_and_calculate_pAtics import clean_and_collect_my_vertices


home = Path.home()
Code_path = os.path.join(home, "Onedrive", "Desktop", "researchProject", "code")
Base_path = os.path.join(Code_path, "o20230614_set3_In3Ca0aa0ar0D0v5Al0Ga3Init1")
Vertices_Path_NEW = os.path.join(Base_path, "vertices_not_cleaned_NEW")
Vertices_Path_OLD = os.path.join(Base_path, "vertices_not_cleaned_OLD_OUTPUT")

# Load the .npy file
clean_and_collect_my_vertices(Vertices_Path_OLD, 10)
plt.show()