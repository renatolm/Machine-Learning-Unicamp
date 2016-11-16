import numpy as np
import pandas as pd
import string
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer

data = load_files('filesk1/')
#print data

categories = pd.read_csv('category.tab').values
#print categories

count_vect = CountVectorizer()
data['data'] = str(data['data']).encode('UTF-8','strict')
print data['data']
data['data'] = str(data['data']).decode('UTF-8','strict')
print data['data']
data['data'] = str(data['data']).translate(string.maketrans("",""), string.punctuation)
data['data'] = count_vect.build_preprocessor()(str(data['data']))
data['data'] = count_vect.build_tokenizer()(str(data['data']))
data['data'] = count_vect.build_analyzer()(str(data['data']))
#print data['data']
