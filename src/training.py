# ### Modulo 3. training.py ➞ treina o modelo ExtraTrees
from modulo import aula3_modulo_mlops as sts
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
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

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MLops_ExtraTrees_Aula04_Final")   

# Exportando os resultados do treinamento
with mlflow.start_run(run_name='training'):
    mlflow.log_params(dict_params)
    mlflow.log_metric('nrms_teste', nrms_teste)
    mlflow.log_metric('r2', r2_teste)
    mlflow.log_metric('mape', mape_teste)
    mlflow.sklearn.log_model(ET, 'model')

    print("Modelo treinado e salvo com sucesso no MLflow.")

print("Modelo treinado e salvo com sucesso.")


