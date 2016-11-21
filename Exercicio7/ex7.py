import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Leitura dos dados dos arquivos csv utilizando o pandas
dados1 = pd.read_csv('serie1.csv')
dados2 = pd.read_csv('serie2.csv')
dados3 = pd.read_csv('serie3.csv')
dados4 = pd.read_csv('serie4.csv')
dados5 = pd.read_csv('serie5.csv')

#Conversao dos dataframes do pandas para arrays do numpy
array1 = dados1.values
array2 = dados2.values
array3 = dados3.values
array4 = dados4.values
array5 = dados5.values

###############################################################################
def meanAndStd(series, n):
	mean = np.mean(series[0:n-1], axis=0)
	std = np.std(series[0:n-1], axis=0)
	return mean, std

################################################################################
#Serie 1
#Calcula a media e desvio padrao da serie 1 considerando os 25% primeiros dados
mean1,std1 = meanAndStd(array1[:,1], array1.shape[0]/4)
tol1 = 3*std1

print "mean serie 1: "+str(mean1)
print "std serie 1: "+str(std1)

for i in range(0,array1.shape[0]):
	if array1[i,1] > (mean1+std1+tol1):
		print "anomalia encontrada em "+str(i)+" valor "+str(array1[i,1])
		anomalia1 = i
		break
	elif array1[i,1] < (mean1-std1-tol1):
		print "anomalia encontrada em "+str(i)+" valor "+str(array1[i,1])
		anomalia1 = i
		break

################################################################################
#Serie 2
#Calcula a media e desvio padrao da serie 2 considerando os 25% primeiros dados
mean2,std2 = meanAndStd(array2[:,1], array2.shape[0]/4)
tol2 = 1.75*std2

print "mean serie 2: "+str(mean2)
print "std serie 2: "+str(std2)

for i in range(0,array2.shape[0]):
	if array2[i,1] > (mean2+std2+tol2):
		print "anomalia encontrada em "+str(i)+" valor "+str(array2[i,1])
		anomalia2 = i
		break
	elif array1[i,1] < (mean2-std2-tol2):
		print "anomalia encontrada em "+str(i)+" valor "+str(array2[i,1])
		anomalia2 = i
		break
	elif array2[i,1] < (mean2+std2-tol2):
		print "anomalia encontrada em "+str(i)+" valor "+str(array2[i,1])
		anomalia2 = i
		break
	elif array2[i,1] > (mean2-std2+tol2):
		print "anomalia encontrada em "+str(i)+" valor "+str(array2[i,1])
		anomalia2 = i
		break

#################################################################################
#Graficos das series com as anomalias indicadas
fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

x_ini_1 = array1.shape[0]/2
y_ini_1 = max(array1[:,1])+10
x_dist_1 = anomalia1 - x_ini_1 - 50
y_dist_1 = array1[anomalia1,1] - y_ini_1

ax0.plot(array1[:,1])
ax0.arrow(x_ini_1, y_ini_1, x_dist_1, y_dist_1, head_width=10, head_length=20, fc='r', ec='r')
ax0.set_title("Serie 1")

x_ini_2 = array2.shape[0]/2
y_ini_2 = max(array2[:,1])+10
x_dist_2 = anomalia2 - x_ini_2 - 50
y_dist_2 = array2[anomalia2,1] - y_ini_2

ax1.plot(array2[:,1])
ax1.arrow(x_ini_2, y_ini_2, x_dist_2, y_dist_2, head_width=10, head_length=20, fc='r', ec='r')
ax1.set_title("Serie 2")

ax2.plot(array3[:,1])
ax2.set_title("Serie 3")

ax3.plot(array4[:,1])
ax3.set_ylim([min(array4[:,1])-2,max(array4[:,1])+2])
ax3.set_title("Serie 4")

ax4.plot(array5[:,1])
ax4.set_title("Serie 5")

plt.tight_layout()
plt.show()
