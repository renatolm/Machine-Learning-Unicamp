import numpy as np 
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 

from sklearn import metrics
from sklearn.cluster import KMeans

#Carregando o conjunto de dados do csv usando o pandas
data = pd.read_csv('cluster-data.csv')

#Conversao para arrays do numpy
array = data.values

#Carregando o conjunto de classes dos dados do csv usando o pandas
data_class = pd.read_csv('cluster-data-class.csv')

#Conversao para arrays do numpy
array_class = data_class.values

#Inicializacao das variaveis que armazenarao o melhor k (numero de clusters)
# e o melhor score interno
internal_k = 0
internal_score = 0

#Loop de escolha do melhor k segundo a metrica interna (Silhouette)
for k in range(2,11):
	
	#Instanciacao do KMeans e avaliacao do score da clusterizacao
	kmeans = KMeans(n_clusters=k, n_init=5, max_iter=1000, init='random').fit(array)
	labels = kmeans.labels_	
	current_score = metrics.silhouette_score(array, labels, metric='euclidean')

	#Escolha do melhor k na avaliacao interna
	if current_score > internal_score:
		internal_score = current_score
		internal_k = k

print "k escolhido segundo a metrica interna: "+str(internal_k)

#Inicializacao das variaveis que armazenarao o melhor k (numero de clusters)
# e o melhor score externo
external_k = 0
external_score = 0

#Loop de escolha do melhor k segundo a metrica interna (Silhouette)
for k in range(2,11):
	
	#Instanciacao do KMeans e avaliacao do score da clusterizacao
	kmeans = KMeans(n_clusters=k, n_init=5, max_iter=1000, init='random').fit(array)
	labels = kmeans.labels_	
	current_score = metrics.adjusted_rand_score(array_class[:,0], labels)

	#Escolha do melhor k na avaliacao interna
	if current_score > external_score:
		external_score = current_score
		external_k = k

print "k escolhido segundo a metrica externa: "+str(external_k)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for row in array:
# 	xs = row[0]
# 	ys = row[1]
# 	zs = row[2]
# 	ax.scatter(xs, ys, zs)

# plt.show()