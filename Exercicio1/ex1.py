import numpy as np 
from sklearn.decomposition import PCA
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale

#Carregar o conjunto de dados do csv usando o pandas
data = pd.read_csv('data1.csv')

#Converter para arrays do numpy (necessario para aplicar o PCA do sklearn)
array = data.values

#Excluindo a ultima coluna da aplicacao do PCA
X = array[:,0:165]
#Y = array[:,166]

pca = PCA(n_components=165)
pca.fit(X)

#print pca.explained_variance_ratio_

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print var1