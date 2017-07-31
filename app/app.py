import csv
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities

### reading csv file
letters = []

csv_file_path = "data\shareholders_letter.csv"
with open(csv_file_path, "r") as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        letters.append(row[3])

### tokenizing letters' texts and removing stopwords
texts = []

tokenizer = RegexpTokenizer(r'\w+')
stoplist = get_stop_words('en')
p_stemmer = PorterStemmer()

for text in letters:
    raw = text.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [text for text in tokens if not text in stoplist]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    texts.append(stemmed_tokens)
