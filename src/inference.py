# # ### Modulo 4. inference.py ➞ aplica inferência no seismic_slice
# import numpy as np
# import matplotlib.pyplot as plt
# import mlflow
# import joblib
# from matplotlib.colors import TwoSlopeNorm
# from modulo import aula3_modulo_mlops as sts

# #Carregando os dados para a inferência
# seismic_slice_clean = np.load("outputs/seismic_slice_clean.npy")
# X = np.load("outputs/X.npy")
# y = np.load("outputs/y.npy")

# # Carregando o modelo treinado e variáveis globais
# ET = joblib.load('outputs/model.pkl')
# sts.ET = ET
# sts.X = X
# sts.y = y

# # Aplicando o modelo ML treinado e faz inferência
# seis_prop_vector, seis_estimated = sts.transfer_to_seismic_scale(dados_sismicos=seismic_slice_clean)

# #Exportando os resultados da inferência
# np.save('outputs/seis_estimated.npy', seis_estimated)

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("MLops_ExtraTrees_Aula04_Final_AutoML")   

# # Plotando os resultados da inferência
# with mlflow.start_run(run_name='inference'):
#     mlflow.log_param('model', 'ExtraTrees')
#     mlflow.log_artifact('outputs/seis_estimated.npy', artifact_path='inference_results')
#     print("Resultados da inferência salvos no MLflow.")

# print("Inferência realizada com sucesso e resultados salvos.")


### inference.py – aplica o modelo treinado do PyCaret aos dados sísmicos

import numpy as np
import pandas as pd
from pycaret.regression import load_model, predict_model
import mlflow

# === 1. Carregando dados sísmicos limpos ===
seismic_slice_clean = np.load("outputs/seismic_slice_clean.npy")

# Convertendo para DataFrame com nomes coerentes
df_seismic = pd.DataFrame(seismic_slice_clean, columns=["X", "Y", "Z", "Propriedade"])

# === 2. Carregando modelo treinado pelo PyCaret ===
model = load_model("outputs/best_model_pycaret")

# === 3. Inferência ===
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("projeto_aula4_mlops")

with mlflow.start_run(run_name="inference_automl"):
    # Primeiro faz a predição
    predictions = predict_model(model, data=df_seismic)

    # Renomeia a coluna de predição
    predictions = predictions.rename(columns={"prediction_label": "Propriedade_Prevista"})

    # Inclui coordenadas + previsão como array 2D
    seis_estimated = predictions[["X", "Y", "Z", "Propriedade_Prevista"]].values
    np.save("outputs/seis_estimated.npy", seis_estimated)

    # Opcional: salvar o DataFrame inteiro com coords + previsões
    predictions.to_csv("outputs/seismic_predictions.csv", index=False)

print("Inferência com PyCaret concluída.")
