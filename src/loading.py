# Bibliotecas

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import aula3_MLOPS_modular as sts
import mlflow

# Carregamento dos dados

# ...existing code...
sim_slice = np.load("C:\\Users\\Igorn\\Desktop\\mlops_project\\sim_slice.npy")  # Dados de treino

seismic_slice = np.load("C:\\Users\\Igorn\\Desktop\\mlops_project\\seismic_slice.npy")  # Dados de teste

seismic_slice_GT = np.load("C:\\Users\\Igorn\\Desktop\\mlops_project\\seismic_slice_GT.npy")  # Dados de teste com Software Comercial

print("Carregado")    
# ...existing code...