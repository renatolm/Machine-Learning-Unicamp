import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer

data = load_files('filesk/')
print data.size
