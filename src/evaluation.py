import numpy as np
import matplotlib.pyplot as plt
import mlflow
import joblib
from matplotlib.colors import TwoSlopeNorm
from modulo import aula3_modulo_mlops as sts

#Dados Treino
sim_clean = np.load("outputs/sim_clean.npy")

#Dados para Inferência
seis_estimado = np.load("outputs/seis_estimado.npy")

#Dados de Referência para a modelagem(Software Comnercial)
seismic_slice_GT = np.load("data/seismic_slice_GT.npy")

print('Loading data completed.')

### Calculo dos Residuos: Dados de Referencia - Dados da Inferência ML
seismic_slice_residual_final = sts.residual_calculator(seismic_GT = seismic_slice_GT, seismic_original = seis_estimado)

### Plot das imagens dos dados originais, dados de inferencia e dados de referencia(Ground Truth)
sts.plot_seismic_slice(sim_clean, title="Slice a profundidade ~5000m dos dados de treino")

sts.plot_seismic_slice(seismic_slice_GT, title="Slice a profundidade ~5000m do Resultado-Referência(software comercial)")

sts.plot_seismic_slice(seis_estimado, title="Slice a profundidade ~5000m da Inferência ML")

sts.plot_seismic_slice(seismic_slice_residual_final, title = "Slice a profundidade ~5000m - Residuo da Inferência")

np.save('outputs/residuos.npy', seismic_slice_residual_final)

print('Plotting completed.')