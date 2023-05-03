import ast

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from scipy.stats import gmean

#======================================================================================================================

df = pd.read_csv('/home/madruga/Downloads/data.csv')

# converter a string '[x, y]' em um array numpy [x, y]
for col in df.columns[:10]:
    df[col] = df[col].apply(lambda x: np.fromstring(x[1:-1], sep=','))


for col in df.columns[:10]:
    df[col] = df[col].apply(lambda x: np.log(np.linalg.norm(x.round(2))))

# calculando a média e a soma das magnitudes linha a linha
df['variancia'] = df[df.columns[:10]].apply(np.var, axis=1)
df['media_magnitudes'] = df[df.columns[:10]].apply(np.mean, axis=1)
df['soma_magnitudes'] = df[df.columns[:10]].apply(np.sum, axis=1)
df['average'] = df[df.columns[:10]].apply(np.average, axis=1)
df['desvio_padrao_magnitudes'] = df[df.columns[:10]].apply(np.std, axis=1)
df['valor_max'] = df[df.columns[:10]].apply(np.max, axis=1)
df['valor_min'] = df[df.columns[:10]].apply(np.min, axis=1)

# calcula a diferenca entre os timestamps
df['dif_timestamp'] = np.log(df['end_timestamp'] - df['start_timestamp'])

# criar series da variavel dependente
y = df['inference']

# criar series da variavel independente
# X = df[['desvio_padrao_magnitudes', 'media_magnitudes', 'valor_max', 'valor_min', 'soma_magnitudes', 'average', 'variancia', 'dif_timestamp']]
X = df[['desvio_padrao_magnitudes', 'media_magnitudes', 'valor_max']]

# dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# criar o modelo de regressao logistica
logistic_model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear')
tree_model = DecisionTreeClassifier()
linear_svc = LinearSVC(C=0.1)
random_forest = RandomForestClassifier(n_estimators=200)
svc = SVC()
bayes_model = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=10)
gradient_model = GradientBoostingClassifier()

# ===============================ESCOLHA DOS HIPERPARAMETROS===========================================================
# cria o dicionário de hiperparâmetros para cada modelo
# logistic_params = {'C': [0.1, 1, 10]}
# tree_params = {'max_depth': [10, 50, 100]}
# linear_svc_params = {'C': [0.1, 1, 10]}
# random_forest_params = {'n_estimators': [10, 100, 200]}
# svc_params = {'C': [0.1, 1, 10]}
# bayes_param = {}
# knn_param = {'n_neighbors': [5, 10, 15]}
# gradient_param = {'learning_rate': [0.1,10, 100, 200], 'n_estimators': [10, 100, 200]}
#
# # cria a lista de modelos e seus respectivos hiperparâmetros
# models = [
#     {'name': 'Logistic Regression', 'estimator': logistic_model, 'params': logistic_params},
#     {'name': 'Decision Tree', 'estimator': tree_model, 'params': tree_params},
#     {'name': 'Linear SVC', 'estimator': linear_svc, 'params': linear_svc_params},
#     {'name': 'Random Forest', 'estimator': random_forest, 'params': random_forest_params},
#     {'name': 'svc', 'estimator': svc, 'params': svc_params},
#     {'name': 'bayes', 'estimator': bayes_model, 'params': bayes_param},
#     {'name': 'knn', 'estimator': knn_model, 'params': knn_param},
#     {'name': 'gradient', 'estimator': gradient_model, 'params': gradient_param}
# ]
#
# # executa o GridSearchCV para cada modelo
# for model in models:
#     print(f"Tuning hyperparameters for {model['name']}")
#     clf = GridSearchCV(model['estimator'], model['params'], cv=5, n_jobs=-1)
#     clf.fit(X_train, y_train)
#     print("Best parameters set found on development set:")
#     print(clf.best_params_)
#     print("Grid scores on development set:")
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print(f"{mean:.3f} (+/-{std * 2:.3f}) for {params}")
#     print()


# ================DETERMINACAO DAS MELHORES FEATURES====================================================
logistic_estimator = LogisticRegression()
decision_tree_estimator = DecisionTreeClassifier()
linear_svc_estimator = LinearSVC()
random_forest_estimator = RandomForestClassifier()
svc_estimator = SVC()
bayes_estimator = GaussianNB()
knn_estimator = KNeighborsClassifier()
gradient_estimator = GradientBoostingClassifier()

# selecionar o número de features que deseja manter
n_features_to_select = 3

# criar o seletor de features RFE
logistic_rfe = RFE(logistic_estimator, n_features_to_select=n_features_to_select)


# ajustar o seletor de features aos dados de treino
logistic_rfe.fit(X_train, y_train)


# selecionar as features que foram mantidas pelo seletor
logistic_selected_features = X_train.columns[logistic_rfe.support_]

print(f"Features selecionadas pelo modelo Logistic Regression: {logistic_selected_features}")

# ===================================================================



# ajustar o modelo com os dados de treino
logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
linear_svc.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
svc.fit(X_train, y_train)
bayes_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
gradient_model.fit(X_train, y_train)


# prever as respostas para o conjunto de teste
y_pred_logistic = logistic_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)
y_pred_linear_svc = linear_svc.predict(X_test)
y_pred_random_forest = random_forest.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_bayes = bayes_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)
y_pred_gradient = gradient_model.predict(X_test)

# calcular a acuracia do modelo
logistic_acc = accuracy_score(y_test, y_pred_logistic.round(2))
tree_acc = accuracy_score(y_test, y_pred_tree.round(2))
svc_acc = accuracy_score(y_test, y_pred_linear_svc.round(2))
random_acc = accuracy_score(y_test, y_pred_random_forest.round(2))
svc_comum_acc = accuracy_score(y_test, y_pred_svc.round(2))
bayes_acc = accuracy_score(y_test, y_pred_bayes.round(2))
knn_acc = accuracy_score(y_test, y_pred_knn.round(2))
gradient_acc = accuracy_score(y_test, y_pred_gradient.round(2))

# criar e treinar modelo
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=1)
clf.fit(X_train, y_train)

# avaliar modelo
score = clf.score(X_test, y_test)

print('Acurácia:', score)
print("OS DADOS DE TREINO FORAM A SOMA DAS MEDIAS DE CADA READ E A DIFERENCA ENTRE OS TIMESTAMPS")
print(f'Acurácia do modelo de logistica: {logistic_acc.round(2)}')
print(f'Acurácia do modelo de decision tree: {tree_acc.round(2)}')
print(f'Acurácia do modelo de SVC linear: {svc_acc.round(2)}')
print(f'Acurácia do modelo de SVC comum: {svc_comum_acc.round(2)}')
print(f'Acurácia do modelo de random forest: {random_acc.round(2)}')
print(f'Acurácia do modelo de bayes: {bayes_acc.round(2)}')
print(f'Acurácia do modelo de knn: {knn_acc.round(2)}')
print(f'Acurácia do modelo de gradient: {gradient_acc.round(2)}')
exit()

# plotar o gráfico de dispersão
# plt.scatter(X_test['valor_max'], y_test, color='blue')
# plt.scatter(X_test['valor_max'], y_pred_logistic, color='red')
#
# # definir o título e os rótulos dos eixos
# plt.title('Gráfico de Dispersão - Resultados do Teste')
# plt.xlabel('Valor Máximo')
# plt.ylabel('Inferência')
#
# # mostrar o gráfico
# plt.show()


#====================================================================================================================================

# CRIANDO MODELOS DE MACHINE LEARNING DE REGRESSAO LINEAR E CLASSIFICACAO
# df = pd.read_csv('/home/madruga/Documentos/Consumo_cerveja.csv', sep=';')
# # AGORA JA COM SKLEARN
#
# # criar series da variavel dependente
# y = df['consumo']
#
# # criar dataframe das variaveis independentes
# X = df[['temp_max', 'chuva', 'fds']]
#
# # dividir os dados em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)
#
# # criar o modelo de regressao linear
# model = LinearRegression()
#
# # utilizar o metodo fit para estimar nosso modelo linear utilizando os dados de treino (X_train, y_train)
# model.fit(X_train, y_train)
#
# # coeficiente de determinacao do modelo de treino
# print(f'R2: {model.score(X_train, y_train).round(2)}')
#
# #gerando previsao para os dados de teste (X_test) utilizando o metodo predict() do objeto model
# y_previsto = model.predict(X_test)
#
# # coeficiente de determinacao do modelo de teste
# print(f'R2: {metrics.r2_score(y_test, y_previsto).round(2)}')
#
# entrada = X_test[0:1]
# print(entrada)
#
# # gerando previsao pontual para a variavel entrada
# print(model.predict(entrada)[0].round(2))
#
# # criando um simulador simples: dadas as variaveis explicativas, qual o consumo de cerveja previsto?
# temp_max = 30.5
# chuva = 12.2
# fds = 1
# entrada = [[temp_max, chuva, fds]]
#
# print(f'Consumo previsto: {model.predict(entrada)[0]}')
#
# # obtendo os coeficientes de regressao (tem_max, chuva, fds)
# print(model.coef_)
#
# print(X.columns)
#
# # criando uma lista com os nomes das variaveis do modelo
# index = ['Intercepto', 'temp_max', 'chuva', 'fds']
#
# # Criando um dataframe para armazenar os coeficientes do modelo
# print(pd.DataFrame(data=np.append(model.intercept_, model.coef_), index=index, columns=['Parametros']))
#
# # analise grafica das previsoes do modelo
#
# # gerando as previsoes do modelo para os dados de treino
# y_previsto_train = model.predict(X_train)
#
#
# #grafico de dispersao entre os valores reais e os valores previstos
# ax = sns.scatterplot(x=y_previsto_train, y=y_train)
# ax.figure.set_size_inches(12, 6)
# ax.set_title('Previsao x Real', fontsize=20)
# ax.set_xlabel('Consumo de Cerveja (Litros) - Previsao', fontsize=16)
# ax.set_ylabel('Consumo de Cerveja (Litros) - Real', fontsize=16)
# plt.show()
#
#
# # obtendo os residuos do modelo
# residuo = y_train - y_previsto_train
# # print(residuo)
#
# # grafico de dispersao entre valor estimado e residuo
# # metodo informal de verificacao da hipotese de variancia constante dos residuos
# ax = sns.scatterplot(x=y_previsto_train, y=residuo)
# ax.figure.set_size_inches(20, 8)
# ax.set_title('Residuos x Previsao', fontsize=20)
# ax.set_xlabel('Consumo de Cerveja (Litros) - Previsao', fontsize=16)
# ax.set_ylabel('Residuos', fontsize=16)
# plt.show()

# =================== COMPARANDO MODELOS ==========================
# df = pd.read_csv('/home/madruga/Documentos/Consumo_cerveja.csv', sep=';')
#
# # criar dataframe das variaveis independentes
# X = df[['temp_max', 'chuva', 'fds']]
#
# # criar series da variavel dependente
# y = df['consumo']
#
# # criar o modelo de regressao linear
# model = LinearRegression()
#
# # # dividir os dados em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)
#
# # utilizar o metodo fit para estimar nosso modelo linear utilizando os dados de treino (X_train, y_train)
# model.fit(X_train, y_train)
#
# # coeficiente de determinacao do modelo de treino
# # print(f'R2 utilizando temperatura maxima: {model.score(X_train, y_train).round(2)}')
#
#
#
# # estimando um novo modelo com a substituicao da variavel explicativa 'temp_max' pela 'temp_media'
# x2 = df[['temp_media', 'chuva', 'fds']]
#
# # criar os datasets de treino e teste
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.3, random_state=2811)
#
# # criar o modelo de regressao linear
# model2 = LinearRegression()
#
# # utilizar o metodo fit para estimar nosso modelo linear utilizando os dados de treino (X2_train, y2_train)
# model2.fit(x2_train, y2_train)
#
# # obtendo o coeficiente de determinacao do novo modelo estimado e comparando com o resultado do modelo anterior
# # print(f'R2 com temperatura media: {model2.score(x2_train, y2_train).round(2)}')
#
# # gerando previsoes para os dados de teste (x_test e x2_test) uutilizando o metodo predict() dos objetos model e model2
# y_previsto = model.predict(X_test)
# y_previsto_2 = model2.predict(x2_test)
#
# # obtendo o coeficiente de determinacao para as previsoes dos dois modelos
# # print(f'R2 com temperatura media: {metrics.r2_score(y2_test, y_previsto_2).round(2)}')
# #
# # print(f'R2 com temperatura maxima: {metrics.r2_score(y_test, y_previsto).round(2)}')
#
#
# ## OUTRAS METRICAS DE REGRESSAO
# # obtendo mettricas para o modelo de temperatura media
# EQM_2 = metrics.mean_squared_error(y2_test, y_previsto_2).round(2)
# REQM_2 = np.sqrt(metrics.mean_squared_error(y2_test, y_previsto_2)).round(2)
# R2_2 = metrics.r2_score(y2_test, y_previsto_2).round(2)
#
# print(pd.DataFrame([EQM_2, REQM_2, R2_2], index=['EQM', 'REQM', 'R2'], columns=['Metricas']))
#
# # obtendo metricas para o modelo de temperatura maxima
# EQM = metrics.mean_squared_error(y_test, y_previsto).round(2)
# REQM = np.sqrt(metrics.mean_squared_error(y_test, y_previsto)).round(2)
# R2 = metrics.r2_score(y_test, y_previsto).round(2)
# # QUANTO MENORES OS VALORES DE EQM E REQM, MELHOR O MODELO
# print(pd.DataFrame([EQM, REQM, R2], index=['EQM', 'REQM', 'R2'], columns=['Metricas']))
#
#
# # SALVANDO E CARREGANDO MODELOS
# # o modulo pickle implementa protocolos binarios para serializacao e desserializacao de objetos python
# # import pickle
# # output = open('modelo_acelerador.pkl', 'wb')
# # pickle.dump(model, output)
# # output.close()
#
#


# print(df.describe().round(2))

# matriz de correlacao
# print(df.corr(method='pearson').round(4))

# plotando a variavel dependente
# fig, ax = plt.subplots(figsize=(20, 6))
# ax = df['consumo'].plot()
# ax.set_title('Consumo de cerveja', fontsize=20)
# ax.set_xlabel('Dias', fontsize=16)
# ax.set_ylabel('Litros', fontsize=16)
# plt.show()
#

# criando boxplot da variavel dependente (y)
# ax = sns.boxplot(data=df['consumo'], width=0.2)
# ax.figure.set_size_inches(12, 6)
# ax.set_title('Boxplot - Consumo de cerveja', fontsize=20)
# ax.set_ylabel('Litros', fontsize=16)
# plt.show()


#boxplot com duas variaveis
#o grafico significa que no fds o consumo de cerveja é maior
# ax = sns.boxplot(y='consumo', x='fds', data=df, width=0.2)
# ax.figure.set_size_inches(12, 6)
# ax.set_title('Boxplot - Consumo de cerveja', fontsize=20)
# ax.set_ylabel('Litros', fontsize=16)
# ax.set_xlabel('Final de Semana', fontsize=16)
# plt.show()


#distribuicao de frequencias
#
# ax = sns.distplot(df['consumo'])
# ax.figure.set_size_inches(12, 6)
# ax.set_title('Distribuicao de Frequencias', fontsize=20)
# ax.set_ylabel('Consumo de Cerveja (Litros)', fontsize=16)
# plt.show()

#pairplot
# ax = sns.pairplot(df, y_vars=['consumo'], x_vars=['temp_min', 'temp_media', 'temp_max', 'chuva', 'fds'], kind='reg')
# ax.fig.suptitle('Pairplot - Consumo de Cerveja', fontsize=20, y=1.05)
#
# plt.show()




# def convert_to_float(val):
#     if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
#         val_list = val[1:-1].split(',')
#
#         val_list = [float(v.strip()) for v in val_list]
#         print(val_list)
#         return val_list
#     else:
#         return val
#
#
# # Carrega os dados em um dataframe pandas
# df = pd.read_csv('/home/madruga/Documentos/Consumo_cerveja.csv', sep=',')
#
# # Divide os dados em features (x) e target (y)
# x = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values
#
# # Divide os dados em treinamento e teste
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# # Cria um modelo de árvore de decisão
# model = DecisionTreeClassifier()
#
# # Treina o modelo com os dados de treinamento
# model.fit(x_train, y_train)
#
# # Faz previsões com os dados de teste
# y_pred = model.predict(x_test)
#
# # Calcula a acurácia do modelo
# accuracy = accuracy_score(y_test, y_pred)
#
# # Exibe a acurácia
# print('Acurácia: {:.2f}%'.format(accuracy * 100))