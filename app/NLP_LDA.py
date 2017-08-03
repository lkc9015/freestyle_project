import csv
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
from sklearn.decomposition import LatentDirichletAllocation

## Read csv file ###
letters = []

csv_file_path = "data\shareholders_letter.csv"
with open(csv_file_path, "r") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for row in reader:
        letters.append(row[3])


### tokenize letters' texts and remove stopwords ###
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

### remove tokens which appear less than five times ###
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 5)

clean_texts = []

for text in texts:
    once_tokens = [word for word in text if word not in tokens_once]
    clean_texts.append(once_tokens)

### Change the texts into Documents Term Matix (dtm) ####
# turn the tokenized documents into a term dictionary
dictionary = corpora.Dictionary(clean_texts)
# convert tokenized documents into a document-term matrix
dtm = [dictionary.doc2bow(text) for text in clean_texts]

### Generate ten topics & ten top words in each topic ###
num_topics = 10
lda = gensim.models.ldamodel.LdaModel(dtm, num_topics=num_topics, id2word = dictionary, passes=10)

for i in  lda.show_topics(num_words=4):
    print (i[0], i[1])
