# ### Modulo 1. loading.py ➞ carrega os dados
from modulo import aula3_modulo_mlops as sts
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mlflow


#Dados Treino
sim_slice = np.load("data/sim_slice.npy")

#Dados para Inferência
seismic_slice = np.load("data/seismic_slice.npy")

#Dados de Referência para a modelagem(Software Comnercial)
seismic_slice_GT = np.load("data/seismic_slice_GT.npy")

print('Loading data completed.')

# ### Modulo 2. cleaning.py ➞ limpeza do sim_slice

sim_data = sts.simulation_data_cleaning(simulation_data = sim_slice, value_to_clean = -99.0)

sim_data = sts.simulation_nan_treatment(simulation = sim_slice, value = 0, method = 'replace')

sim_data, seis_cube = sts.depth_signal_checking(simulation_data=sim_slice, seismic_data=seismic_slice)

np.save('outputs/sim_clean.npy', sim_data)

np.save('outputs/seismic_slice_clean.npy', seismic_slice)\

np.save('outputs/seismic_slice_GT_clean.npy', seismic_slice_GT)

print('Data cleaning completed.')