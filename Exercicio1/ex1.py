import numpy as np 
from sklearn.decomposition import PCA
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn import metrics

#Carregar o conjunto de dados do csv usando o pandas
data = pd.read_csv('data1.csv')

#Converter para arrays do numpy (necessario para aplicar o PCA do sklearn)
array = data.values

#Excluindo a ultima coluna da aplicacao do PCA
X = array[:,0:165]
Y = array[:,166]

#Aplicando o PCA com todas as 165 componentes
pca = PCA(n_components=165)
pca.fit(X)

#Verificando a variancia acumulada
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print var

#Selecionando o numero de componentes que mantem a variancia em 80% ----- Resposta da pergunta 1
pca = PCA(n_components=12)
X_transf = pca.fit_transform(X)

#Aplicando a regressao logistica aos dados de treino do conjunto de dados original
model = LogisticRegression()
model = model.fit(X[:200,:],Y[:200])

#Aplicando a regressao logistica aos dados de treino do conjunto de dados transformado
model_transf = LogisticRegression()
model_transf = model_transf.fit(X_transf[:200,:],Y[:200])

#Fazendo a classificacao dos dados de teste do conjunto de dados original
predicted = model.predict(X[200:])

#Fazendo a classificacao dos dados de teste do conjunto de dados transformado
predicted_transf = model_transf.predict(X_transf[200:])

#Verificando a acuracia das classificacoes ----- Resposta da pergunta 2
print "Acuracia da regressao logistica no conjunto de dados original: "+str(metrics.accuracy_score(Y[200:], predicted))
print "Acuracia da regressao logistica no conjunto de dados transformado: "+str(metrics.accuracy_score(Y[200:], predicted_transf))


#Aplicando o LDA aos dados de treino do conjunto de dados original
model_LDA = LDA()
model_LDA = model_LDA.fit(X[:200],Y[:200])

#Aplicando o LDA aos dados de treino do conjunto de dados transformado
model_LDA_transf = LDA()
model_LDA_transf = model_LDA_transf.fit(X_transf[:200],Y[:200])

#Fazendo a classificacao dos dados de teste do conjunto de dados original
predicted_LDA = model_LDA.predict(X[200:])

#Fazendo a classificacao dos dados de teste do conjunto de dados transformado
predicted_LDA_transf = model_LDA_transf.predict(X_transf[200:])

#Verificando a acuracia das classificacoes ----- Resposta da pergunta 3
print "Acuracia do LDA no conjunto de dados original: "+str(metrics.accuracy_score(Y[200:], predicted_LDA))
print "Acuracia do LDA no conjunto de dados transformado: "+str(metrics.accuracy_score(Y[200:], predicted_LDA_transf))