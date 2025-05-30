#!/usr/bin/env python
# coding: utf-8

# ## Desafio Aula 3:

# ### 1. Modularização para organizar no DVC
# ### 2. Uso do AutoML para automatizar o emprego dos modelos de Machine Learning utilizando  - Pycaret
# ### 3. Criação de uma Pipeline de ML no DVC

# ## Conceitos-Chave
# 
# ### Reprodutibilidade: garantir que os experimentos sejam executáveis a qualquer momento, com os mesmos dados e parâmetros.
# 
# ### Versionamento de dados: aplicar o mesmo controle de versão que temos no código (Git) também aos dados.
# 
# ### Pipelines: criação de etapas encadeadas com dependências rastreáveis.

# ## Desafio Aula 2 (última aula):

# ### Atuando como consultor para uma empresa, a mesma lhe forneceu um código legado de um projeto que não foi para frente com o time de analytics deles.
# ### A empresa é da área de óleo e gás e trabalha mapeando áreas com potencial para explorar.
# ### O projeto deles trata de tentar aumentar a granularidade(resolução) de um conjunto de dados inicial para um conjunto de dados final com "melhor resolução" que permita um mapeamento melhor.
# ### A empresa trabalha com um software comercial que produz resultados razoáveis, mas que é uma caixa preta e o time de negócios da empresa agora resolveu criar suas próprias soluções para ter mais controle e não precisar pagar mais a licença desse software e automatizar os processos.
# ### A empresa lhe forneceu os dados de treino, e os dados de inferência. Ambos em estrutura numpy array com coordenadas X,Y,Z,Propriedade(target).
# ### A empresa também lhe forneceu os dados do resultado gerado por eles com o software comercial, com o mesmo tipo de estrutura dos dados de treino e de inferencia, para que você compare com a solução criada por você.
# ### Cabe a você realizar experimentos novos que melhorem (em relação à solução do software comercial).
# ### Repare que a solução atual que já consta no código legado claramente apresenta artefatos estranhos, explore isso.

# ### Modulo 1. loading.py ➞ carrega os dados

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import modulo.aula3_modulo_mlops as sts
import mlflow

#Dados Treino
sim_slice = np.load("sim_slice.npy")

#Dados para Inferência
seismic_slice = np.load("seismic_slice.npy")

#Dados de Referência para a modelagem(Software Comnercial)
seismic_slice_GT = np.load("seismic_slice_GT.npy")


# ### Modulo 2. cleaning.py ➞ limpeza do sim_slice

# In[8]:


sim_data = sts.simulation_data_cleaning(simulation_data = sim_slice, value_to_clean = -99.0)

sim_data = sts.simulation_nan_treatment(simulation = sim_slice, value = 0, method = 'replace')

sim_data, seis_cube = sts.depth_signal_checking(simulation_data=sim_slice, seismic_data=seismic_slice)


# ### Modulo 3. training.py ➞ treina o modelo ExtraTrees

# In[12]:


sim_estimado, y, nrms_teste, r2_teste, mape_teste, dict_params, ET = sts.ML_model_evaluation(dados_simulacao=sim_slice, proporcao_treino=0.75)


# ### Modulo 4. inference.py ➞ aplica inferência no seismic_slice

# In[13]:


seis_prop_vector, seis_estimated = sts.transfer_to_seismic_scale(dados_sismicos=seismic_slice)


# ### Modulo 5. evaluation.py ➞ compara com seismic_slice_GT

# In[15]:


### Calculo dos Residuos: Dados de Referencia - Dados da Inferência ML
seismic_slice_residual_final = sts.residual_calculator(seismic_GT = seismic_slice_GT, seismic_original = seismic_slice)


# In[ ]:


### Plot dos histogramas com as distribuições dos dados originais, dados de inferencia e dados de referencia(Ground Truth)
sts.plot_simulation_distribution(sim_slice, bins=35)

sts.plot_simulation_distribution(seis_estimated, bins=35)

sts.plot_simulation_distribution(seismic_slice_GT, bins=35)

sts.plot_simulation_distribution(seismic_slice_residual_final, bins=35)


# In[17]:


### Plot das imagens dos dados originais, dados de inferencia e dados de referencia(Ground Truth)
sts.plot_seismic_slice(sim_slice, title="Slice a profundidade ~5000m dos dados de treino")

sts.plot_seismic_slice(seismic_slice_GT, title="Slice a profundidade ~5000m do Resultado-Referência(software comercial)")

sts.plot_seismic_slice(seis_estimated, title="Slice a profundidade ~5000m da Inferência ML")

sts.plot_seismic_slice(seismic_slice_residual_final, title = "Slice a profundidade ~5000m - Residuo da Inferência")


# ### MLFLOW Tracking

# In[21]:


tuple_1 = ["nrms_teste", "r2_teste", "mape_teste"]
   
metrics_tuple = [64.27, 0.58, 71.8]

dict_metrics = dict(zip(tuple_1, metrics_tuple))
dict_metrics


# In[22]:


mlflow.set_registry_uri("http://127.0.0.1:5000")


# In[23]:


mlflow.set_experiment("Aula2 Experimento 3")


# In[24]:


with mlflow.start_run(run_name="Aula2 MLOps Exp 3"):
    mlflow.log_params(dict_params)
    mlflow.log_metrics(dict_metrics)
    mlflow.sklearn.log_model(ET, "Base Line Model ExtraTrees")


# In[25]:


# with mlflow.start_run(run_name="SimToSeis_ET_Model"):
    
#     mlflow.log_param("train_ratio", proporcao_treino)

#     modelo, X, y, nrms, r2, mape = treinar_modelo(sim_data, proporcao_treino)

#     mlflow.log_metric("NRMS", nrms)
#     mlflow.log_metric("R2", r2)
#     mlflow.log_metric("MAPE", mape)

#     mlflow.sklearn.log_model(modelo, "modelo_extra_trees")

#     seis_estimado = inferir_propriedade(modelo, seismic_slice)
#     np.save("seis_estimado.npy", seis_estimado)
#     mlflow.log_artifact("seis_estimado.npy")

#     residuos = calcular_residuos(seismic_slice_GT, seis_estimado)
#     np.save("residuos.npy", residuos)
#     mlflow.log_artifact("residuos.npy")

#     print(f"NRMS: {nrms:.2f}%, R2: {r2:.2f}, MAPE: {mape:.2f}")
#     print("Run registrada com MLflow!")

#     plotar_histograma(seis_estimado, titulo="Inferência Sísmica - Histograma")
#     plotar_histograma(residuos, titulo="Resíduos - Histograma")


# ### FIM
