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
    next(reader)
    for row in reader:
        letters.append(row[3])

### tokenizing letters' texts and removing stopwords
texts = []

tokenizer = RegexpTokenizer(r'\w+')
stoplist = get_stop_words('en')
stemmer = PorterStemmer()

for text in letters:
    raw = text.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [text for text in tokens if text not in stoplist]
    stemmed_tokens = [stemmer.stem(text) for text in stopped_tokens]
    texts.append(stemmed_tokens)

## removing tokens which appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)

clear_texts = []

for text in texts:
    once_tokens = [word for word in text if word not in tokens_once]
    clear_texts.append(once_tokens)
