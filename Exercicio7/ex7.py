import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math

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

print "\n"

################################################################################
#Serie 2
#Calcula a media e desvio padrao da serie 2 considerando os 25% primeiros dados
mean2,std2 = meanAndStd(array2[:,1], array2.shape[0]/4)
tol_mean2 = 0.05*mean2

print "mean serie 2: "+str(mean2)
print "std serie 2: "+str(std2)
print "mean + tol serie 2: "+str(mean2+tol_mean2)
print "mean - tol serie 2: "+str(mean2-tol_mean2)

chunk2 = 1
for i in np.arange(0,array2.shape[0],10):
	if np.mean(array2[0:chunk2,1]) > (mean2+tol_mean2):
		chunk2 = chunk2+10
	elif np.mean(array2[0:chunk2,1]) < (mean2-tol_mean2):
		chunk2 = chunk2+10
	elif (np.mean(array2[0:3*chunk2,1]) < (mean2+tol_mean2)) and (np.mean(array2[0:3*chunk2,1]) > (mean2-tol_mean2)):
		break
	else:
		chunk2 = chunk2+10

print "chunk size: "+str(chunk2)

n_chunks = int(math.floor(array2.shape[0]/chunk2))

for i in range(0,n_chunks):
	ini = chunk2*i
	end = chunk2*(i+1)

	if np.mean(array2[ini:end,1]) > (mean2+tol_mean2):
		print "anomalia encontrada em torno de "+str(ini)+" media "+str(np.mean(array2[ini:end,1]))
		anomalia2 = int(math.floor((ini+end)/2))
		break
	elif np.mean(array2[ini:end,1]) < (mean2-tol_mean2):
		print "anomalia encontrada em torno de "+str(ini)+" media "+str(np.mean(array2[ini:end,1]))
		anomalia2 = int(math.floor((ini+end)/2))
		break

print "\n"

################################################################################
#Serie 3
#Calcula a media e desvio padrao da serie 3 considerando os 25% primeiros dados
mean3,std3 = meanAndStd(array3[:,1], array3.shape[0]/4)
tol_mean3 = 0.05*mean3

print "mean serie 3: "+str(mean3)
print "std serie 3: "+str(std3)
print "mean + tol serie 3: "+str(mean3+tol_mean3)
print "mean - tol serie 3: "+str(mean3-tol_mean3)

chunk3 = 1
for i in np.arange(0,array3.shape[0],10):
	if np.mean(array3[0:chunk3,1]) > (mean3+tol_mean3):
		chunk3 = chunk3+10
	elif np.mean(array3[0:chunk3,1]) < (mean3-tol_mean3):
		chunk3 = chunk3+10
	elif (np.mean(array3[0:3*chunk3,1]) < (mean3+tol_mean3)) and (np.mean(array3[0:3*chunk3,1]) > (mean3-tol_mean3)):
		break
	else:
		chunk3 = chunk3+10

print "chunk size: "+str(chunk3)

n_chunks = int(math.floor(array3.shape[0]/chunk3))

for i in range(0,n_chunks):
	ini = chunk3*i
	end = chunk3*(i+1)

	if np.mean(array3[ini:end,1]) > (mean3+tol_mean3):
		print "anomalia encontrada em torno de "+str(ini)+" media "+str(np.mean(array3[ini:end,1]))
		anomalia3 = int(math.floor((ini+end)/2))
		break
	elif np.mean(array3[ini:end,1]) < (mean3-tol_mean3):
		print "anomalia encontrada em torno de "+str(ini)+" media "+str(np.mean(array3[ini:end,1]))
		anomalia3 = int(math.floor((ini+end)/2))
		break

print "\n"

################################################################################
#Serie 4
#Calcula a media e desvio padrao da serie 4 considerando os 25% primeiros dados
mean4,std4 = meanAndStd(array4[:,1], array4.shape[0]/4)
tol_mean4 = 0.01*mean4

print "mean serie 4: "+str(mean4)
print "std serie 4: "+str(std4)
print "mean + tol serie 4: "+str(mean4+tol_mean4)
print "mean - tol serie 4: "+str(mean4-tol_mean4)

chunk4 = 1
for i in np.arange(0,array4.shape[0],10):
	if np.mean(array4[0:chunk4,1]) > (mean4+tol_mean4):
		chunk4 = chunk4+10
	elif np.mean(array4[0:chunk4,1]) < (mean4-tol_mean4):
		chunk4 = chunk4+10
	elif (np.mean(array4[0:3*chunk4,1]) < (mean4+tol_mean4)) and (np.mean(array4[0:3*chunk4,1]) > (mean4-tol_mean4)):
		print "mean chunk size: "+str(np.mean(array4[0:chunk4,1]))
		print "mean 2xchunk size: "+str(np.mean(array4[0:2*chunk4,1]))
		print "mean 3xchunk size: "+str(np.mean(array4[0:3*chunk4,1]))
		print "mean next chunk: "+str(np.mean(array4[chunk4:2*chunk4,1]))
		break
	else:
		chunk4 = chunk4+10

print "chunk size: "+str(chunk4)

n_chunks = int(math.floor(array4.shape[0]/chunk4))

for i in range(0,n_chunks):
	ini = chunk4*i
	end = chunk4*(i+1)

	if np.mean(array4[ini:end,1]) > (mean4+tol_mean4):
		print "anomalia encontrada em torno de "+str(ini)+" media "+str(np.mean(array4[ini:end,1]))
		anomalia4 = int(math.floor((ini+end)/2))
		break
	elif np.mean(array4[ini:end,1]) < (mean4-tol_mean4):
		print "anomalia encontrada em torno de "+str(ini)+" media "+str(np.mean(array4[ini:end,1]))
		anomalia4 = int(math.floor((ini+end)/2))
		break

print "\n"

#################################################################################
#Graficos das series com as anomalias indicadas
fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

#Grafico serie 1
x_ini_1 = array1.shape[0]/2
y_ini_1 = max(array1[:,1])+10
x_dist_1 = anomalia1 - x_ini_1 - 50
y_dist_1 = array1[anomalia1,1] - y_ini_1

ax0.plot(array1[:,1])
ax0.arrow(x_ini_1, y_ini_1, x_dist_1, y_dist_1, head_width=10, head_length=20, fc='r', ec='r')
ax0.set_title("Serie 1")

#Grafico serie 2
x_ini_2 = array2.shape[0]/2
y_ini_2 = max(array2[:,1])+10
x_dist_2 = anomalia2 - x_ini_2 - 50
y_dist_2 = array2[anomalia2,1] - y_ini_2

ax1.plot(array2[:,1])
ax1.arrow(x_ini_2, y_ini_2, x_dist_2, y_dist_2, head_width=10, head_length=20, fc='r', ec='r')
ax1.set_title("Serie 2")

#Grafico serie 3
x_ini_3 = array3.shape[0]/2
y_ini_3 = max(array3[:,1])+10
x_dist_3 = anomalia3 - x_ini_3 - 50
y_dist_3 = array3[anomalia3,1] - y_ini_3

ax2.plot(array3[:,1])
ax2.arrow(x_ini_3, y_ini_3, x_dist_3, y_dist_3, head_width=10, head_length=20, fc='r', ec='r')
ax2.set_title("Serie 3")

#Grafico serie 4
x_ini_4 = array4.shape[0]/2
y_ini_4 = max(array4[:,1])+10
x_dist_4 = anomalia4 - x_ini_4 - 50
y_dist_4 = array4[anomalia4,1] - y_ini_4

ax3.plot(array4[:,1])
ax3.arrow(x_ini_4, y_ini_4, x_dist_4, y_dist_4, head_width=1, head_length=2, fc='r', ec='r')
ax3.set_ylim([min(array4[:,1])-2,max(array4[:,1])+2])
ax3.set_title("Serie 4")

#Grafico serie 5
ax4.plot(array5[:,1])
ax4.set_title("Serie 5")

plt.tight_layout()
plt.show()
