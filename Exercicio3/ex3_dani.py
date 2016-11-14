# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 23:52:58 2016

@author: daniel
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as nnet
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import GradientBoostingClassifier as gbm

#lendo os dados e as classes
data = pd.read_table('secom.data', header=None, delim_whitespace=True)
classes = pd.read_table('secom_labels.data', header=None, 
                        delim_whitespace=True,usecols=[0])

#fazendo as medias dos valores
mean = data.mean().values

#substituindo a media nos valores faltantes
for i in data.columns:
    data.loc[data.isnull().iloc[:,i]==True, i] = mean[i];data

#arrays do numpy
data = data.values
classes = classes.iloc[:,0].values

#parametros a serem avaliados
p_knn = {'n_neighbors':[1, 5, 11, 15, 21, 25]}
p_svm = {'C' : [2**(-5), 2**(0), 2**(5), 2**(10)],
        'gamma' : [2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)]}
p_nn = {'hidden_layer_sizes' :[10, 20, 30, 40]}
p_rf = {'n_estimators':[100, 200, 300, 400],
                 'max_features':[10, 15, 20, 25]}
p_gbm = {'learning_rate' :[0.1, 0.05],'n_estimators':[30, 70, 100]}

#inicializando a acuracia
acc_mean = [0]*5

#utilizando o 5-fold
kfold = StratifiedKFold(n_splits=5)

for tr, te in kfold.split(data, classes):

    #compondo o conjunto de treino e conjunto de teste    
    data_tr = data[tr]
    data_te = data[te]
    classes_tr = classes[tr]
    classes_te = classes[te]
    
    #padronizacao dos dados baseados no conjunto de treinamento
    scale = StandardScaler().fit(data_tr)
    data_tr = scale.transform(data_tr)
    data_te = scale.transform(data_te)
    
    #KNN
    
    #PCA
    pca = PCA(0.8)
    pca.fit(data_tr)
    data_tr_pca = pca.transform(data_tr)
    data_te_pca = pca.transform(data_te)
    
    #estimando o parametro em 3 fold
    grid = GSCV(KNN(), p_knn)
    grid.fit(data_tr_pca, classes_tr)
    
    #acuracia
    knn = KNN(n_neighbors=grid.best_params_['n_neighbors'])
    knn.fit(data_tr_pca, classes_tr)
    acc = knn.score(data_te_pca, classes_te)
    acc_mean[0] += acc/5
    
    #SVM 
    
    #estimando o parametro em 3 fold
    grid = GSCV(SVC(), p_svm)
    grid.fit(data_tr, classes_tr)
    
    #acuracia
    svm = SVC(C=grid.best_params_['C'],gamma=grid.best_params_['gamma'],
              kernel='rbf')
    svm.fit(data_tr_pca, classes_tr)
    acc = svm.score(data_te_pca, classes_te)
    acc_mean[1] += acc/5
    
    #Rede neural    
    
    #estimando o parametro em 3 fold
    grid = GSCV(nnet(solver='lbfgs'), p_nn)
    grid.fit(data_tr, classes_tr)
    
    #acuracia
    nn = nnet(hidden_layer_sizes = grid.best_params_['hidden_layer_sizes'], solver = 'lbfgs')
    nn.fit(data_tr_pca, classes_tr)
    acc = nn.score(data_te_pca, classes_te)
    acc_mean[2] += acc/5

    #Ramdon forest    
    
    #estimando o parametro em 3 fold
    grid = GSCV(rfc(), p_rf)
    grid.fit(data_tr, classes_tr)
    
    #acuracia
    rf = rfc(n_estimators = grid.best_params_ ['n_estimators'],
             max_features = grid.best_params_ ['max_features'])
    rf.fit(data_tr_pca, classes_tr)
    acc = rf.score(data_te_pca, classes_te)
    acc_mean[3] += acc/5

    #Gradient Boosting Machine    
    
    #estimando o parametro em 3 fold
    grid = GSCV(gbm(max_depth=5), p_gbm)
    grid.fit(data_tr, classes_tr)
    
    #acuracia
    gb = gbm(learning_rate = grid.best_params_ ['learning_rate' ],
           n_estimators = grid.best_params_ ['n_estimators'])
    gb.fit(data_tr_pca, classes_tr)
    acc = gb.score(data_te_pca, classes_te)
    acc_mean[4] += acc/5
    
for i in acc_mean:
    acc_mean[i] = np.round(acc_mean[i]*100,2)
    
#Mostrando a acuracia de cada metodo
print 'Acuracias de cada metodo:'
print 'K-Nearest Neighbors - '+str(acc_mean[0])+' %'
print 'Support Vector Machine - '+str(acc_mean[1])+' %'
print 'Neural Network - '+str(acc_mean[2])+' %'
print 'Ramdon Forest - '+str(acc_mean[3])+' %'
print 'Gradient Boosting Machine - '+str(acc_mean[4])+' %'

