import csv
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
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
        full_contents.append(row)
        letters.append(row[3])

#### tokenizing letters' texts and removing stopwords
texts = []

tokenizer = RegexpTokenizer(r'\w+')
stoplist = get_stop_words('en')
stemmer = PorterStemmer()

for text in letters:
    lower = text.lower()
    tokens = tokenizer.tokenize(lower)
    stopped_tokens = [text for text in tokens if text not in stoplist]
    stemmed_tokens = [stemmer.stem(text) for text in stopped_tokens]
    texts.append(stemmed_tokens)

## removing tokens which appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 3)

clean_texts = []

for text in texts:
    once_tokens = [word for word in text if word not in tokens_once]
    clean_texts.append(once_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(clean_texts)

# convert tokenized documents into a document-term matrix
dtm = [dictionary.doc2bow(text) for text in clean_texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(dtm, num_topics=10, id2word = dictionary, passes=20)
