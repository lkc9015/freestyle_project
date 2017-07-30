import csv
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
