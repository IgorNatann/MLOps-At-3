# ### Modulo 4. inference.py ➞ aplica inferência no seismic_slice
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import joblib
from matplotlib.colors import TwoSlopeNorm
from modulo import aula3_modulo_mlops as sts

#Carregando os dados para a inferência
seismic_slice_clean = np.load("outputs/seismic_slice_clean.npy")
X = np.load("outputs/X.npy")
y = np.load("outputs/y.npy")

# Carregando o modelo treinado e variáveis globais
ET = joblib.load('outputs/model.pkl')
sts.ET = ET
sts.X = X
sts.y = y

# Aplicando o modelo ML treinado e faz inferência
seis_prop_vector, seis_estimated = sts.transfer_to_seismic_scale(dados_sismicos=seismic_slice_clean)

#Exportando os resultados da inferência
np.save('outputs/seis_estimated.npy', seis_estimated)

print("Inferência realizada com sucesso e resultados salvos.")