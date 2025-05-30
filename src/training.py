# ### Modulo 3. training.py ➞ treina o modelo ExtraTrees
from modulo import aula3_modulo_mlops as sts
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import joblib
from matplotlib.colors import TwoSlopeNorm

# Carrega dados de treino
sim_clean = np.load("outputs/sim_clean.npy")

#Treina o modelo nos dados de treino
sim_estimado, y, nrms_teste, r2_teste, mape_teste, dict_params, ET, X = sts.ML_model_evaluation(dados_simulacao = sim_clean, proporcao_treino = 0.75)

#Exportando o modelo treinado
joblib.dump(ET, 'outputs/model.pkl')

np.save('outputs/X.npy', X)
np.save('outputs/y.npy', y)

print("Modelo treinado e salvo com sucesso.")


