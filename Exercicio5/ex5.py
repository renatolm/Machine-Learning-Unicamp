import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

#############################################################################
#Carregando o conjunto de dados de treino do csv usando o pandas
data = pd.read_csv('train.csv', header=None)

#Separando os valores a serem estimados do resto dos dados
train_Y = data.pop(0)

#Separando os dados numericos dos categoricos
numericos = data.select_dtypes(include=['int64']).columns
categoricos = data.select_dtypes(include=['object']).columns

#Exibindo quais colunas contem cada tipo de dado
print "numericos: "+str(numericos.values)
print "categoricos: "+str(categoricos.values)

#Convertendo os dados categoricos para labels numericos
for column in categoricos:
	data[column] = pd.Categorical.from_array(data[column]).labels

#############################################################################
#Eliminando as colunas de dados numericos com variancia menor do que 1
numericos_new = []

for column in numericos:
	if data[column].var() < 1:
		data.pop(column)
	else:
		numericos_new.append(column)

numericos = pd.Index(numericos_new)
print "numericos restantes: "+str(numericos.values)

#############################################################################
#Aplicando escala e o PCA nos dados numericos de treino
numericos_array = data[numericos].values

numericos_array_scaled = preprocessing.scale(numericos_array)

pca = PCA(0.8)
numericos_array = pca.fit_transform(numericos_array_scaled)

print "componentes restantes apos o pca: "+str(pca.n_components_)

#############################################################################
#Juntando os dados de treino numericos e categoricos
train_X = np.concatenate((numericos_array, data[categoricos].values), axis=1)
print train_X.shape

#############################################################################
#Carregando o conjunto de dados de teste do csv usando o pandas
data_test = pd.read_csv('test.csv', header=None)

print data_test.describe

#Separando os valores a serem estimados do resto dos dados
test_Y = data_test.pop(0)

#Convertendo os dados categoricos para labels numericos
for column in categoricos:
	data_test[column] = pd.Categorical.from_array(data_test[column]).labels

#############################################################################
#Aplicando escala e o PCA nos dados numericos de teste
numericos_array_test = data_test[numericos].values

numericos_array_scaled_test = preprocessing.scale(numericos_array_test)

pca = PCA(n_components=10)
numericos_array_test = pca.fit_transform(numericos_array_scaled_test)

#############################################################################
#Juntando os dados de teste numericos e categoricos
test_X = np.concatenate((numericos_array_test, data_test[categoricos].values), axis=1)
print test_X.shape
