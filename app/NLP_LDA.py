import csv
import sklearn.feature_extraction.text as text
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

## Read csv file ###
letters = []

csv_file_path = "data\shareholders_letter.csv"
with open(csv_file_path, "r") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for row in reader:
        letters.append(row[3])


#### Change the texts into Documents Term Matix (dtm) ####
# remove stopwords and words that appear less than five times
vectorizer = text.CountVectorizer(input = letters,  stop_words = 'english', min_df = 10)
# create dtm and convert it into an array
dtm = vectorizer.fit_transform(letters).toarray()
# list of words and change it to an array
vocab = np.array(vectorizer.get_feature_names())


### Generate ten topics & ten top words in each topic ###
n_topics = 10
n_top_words = 10
# classifier
lda = LatentDirichletAllocation(n_topics = n_topics, random_state=1)
doctopic = lda.fit_transform(dtm)


### Print out topic words ###
topic_words = []
for topic in lda.components_:
    word_idx = np.argsort(topic)[::-1][0:n_top_words]
    topic_words.append([vocab[i] for i in word_idx])

for topic in range(len(topic_words)):
    print(" + Topic {}: {}".format(topic, ' '.join(topic_words[topic])))
