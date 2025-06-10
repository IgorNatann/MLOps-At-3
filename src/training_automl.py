# ### Modulo 3. training.py ➞ treina o modelo ExtraTrees
from modulo import aula3_modulo_mlops as sts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib
from matplotlib.colors import TwoSlopeNorm
from pycaret.regression import *

# Carrega dados de treino
sim_clean = np.load("outputs/sim_clean.npy")

#Converte em DataFrame para usar com PyCaret
df = pd.DataFrame(sim_clean, columns=["X", "Y", "Z", "Propriedade"])

#Tracking do modelo
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MLops_ExtraTrees_Aula04_AutoML")   

#Setup do PyCaret
s = setup(data=df, target='Propriedade', session_id=123, log_experiment=True, experiment_name='MLops_ExtraTrees_Aula04_Final', log_plots=True)
print("Modelo treinado e salvo com sucesso.")

# Compara todos os modelos disponíveis
best_model = compare_models()

# Exporta o melhor modelo
save_model(best_model, 'outputs/best_model_pycaret')

print("AutoML com PyCaret salvo com sucesso.")
# Note: This code is a direct translation of the original training script, which trains an ExtraTrees model on seismic data.