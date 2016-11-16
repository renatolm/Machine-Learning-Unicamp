import numpy as np
import pandas as pd
import string
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk import PorterStemmer

def removeNonAscii(s):
	return re.sub(r'\\x\w{2}','',s)

data = load_files('filesk1/')

count_vect = CountVectorizer(stop_words='english')

data['data'] = count_vect.build_preprocessor()(str(data['data']))
data['data'] = removeNonAscii(data['data'])
data['data'] = count_vect.build_tokenizer()(str(data['data']))
data['data'] = count_vect.build_analyzer()(str(data['data']))

stemmed_data = []
for word in data['data']:
	word = PorterStemmer().stem(word)
	stemmed_data.append(word)

data['data'] = stemmed_data

print data['data']
