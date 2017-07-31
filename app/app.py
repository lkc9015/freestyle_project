import csv
import os
import numpy as np
import sklearn.feature_extraction.text as text
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

letters = []

csv_file_path = "data\shareholders_letter.csv"
with open(csv_file_path, "r") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        letters.append(dict(row))

print(letters[1:2])
texts = []

#for text in letters["Text"]:
#    raw = text.lower()
#    tokens = tokenizer.tokenize(raw)
#    stopped_tokens = [text for text in tokens if not text in en_stop]
#    stemmed_tokens = [p_stemmer.stem(text) for text in stopped_tokens]
#    texts.append(stemmed_tokens)




#vectorizer = text.CountVectorizer(input='letters', stop_words='english', min_df=20)
#dtm = vectorizer.fit_transform(letters["Text"]).toarray()
#vocab = np.array(vectorizer.get_feature_names())
#print(dtm.shape)

#from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
#from nltk.stem.porter import PorterStemmer
#from gensim import corpora, models
#import gensim
#
#letters = []
#
#csv_file_path = "data\shareholders_letter.csv"
#with open(csv_file_path, "r") as csv_file:
#    reader = csv.DictReader(csv_file)
#    for row in reader:
#        letters.append(dict(row))
#
#print (letters[:5])
