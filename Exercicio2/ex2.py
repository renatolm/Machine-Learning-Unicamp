import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedKFold

#Carregando o conjunto de dados do csv usando o pandas
data = pd.read_csv('data1.csv')

#Conversão para arrays do numpy
array = data.values

#Separando os dados em atributos e classes
X = array[:,0:165]
Y = array[:,166]

#Lista de hiperparâmetros no formato [C,gamma]
hyperparameters = [[2**(-5), 2**(-15)],[2**(-2), 2**(-10)],[2**0, 2**(-5)],[2**2, 2**0],[2**5, 2**5]]

#Aplicação do StratifiedKFold no loop externo (5 folds)
external = StratifiedKFold(n_splits=5)
