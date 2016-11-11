import numpy as np 
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

#Carregando o conjunto de dados do csv usando o pandas
data = pd.read_csv('train.csv', header=None)

#Conversao para arrays do numpy
#array = data.values

print data.describe

#print data.isnull().sum()

#print data.dtypes

train_Y = data.pop(0)

numericos = data.select_dtypes(include=['int64'])
categoricos = data.select_dtypes(include=['object'])

print numericos.shape
print categoricos.shape

print categoricos.head(5)

array = categoricos.values

number = LabelEncoder()
teste = number.fit_transform(array.astype('str'))

print categoricos.var()

#selector = VarianceThreshold()
#numericos_mod = selector.fit_transform(numericos)

#print numericos_mod.shape
