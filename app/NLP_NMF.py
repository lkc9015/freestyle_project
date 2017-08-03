import csv
import numpy as np
import sklearn.feature_extraction.text as text
from sklearn import decomposition

## Read CSV file
letters = []

csv_file_path = "data\shareholders_letter.csv"
with open(csv_file_path, "r") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for row in reader:
        letters.append(row[3])

#### Change the texts into Documents Term Matix (dtm) ####
# remove stopwords and words that appear less than five times
vectorizer = text.CountVectorizer(input='letters', stop_words='english', min_df=5)
# create dtm and convert it into an array
dtm = vectorizer.fit_transform(letters).toarray()
# list of words and change it to an array
vocab = np.array(vectorizer.get_feature_names())

### Generate ten topics & ten top words in each topic ###
num_topics = 10
num_top_words = 10
# classifier
nmf = decomposition.NMF(n_components=num_topics, random_state=1)
doctopic = nmf.fit_transform(dtm)

### Print out topic words ###
topic_words = []
for topic in nmf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])

for topic in range(len(topic_words)):
    print(" + Topic {}: {}".format(topic, ' '.join(topic_words[topic])))



#nmf_path = "data\NLP_NMF_topics.csv"
#with open(nmf_path, "w") as csv_file:
#    writer = csv.DictWriter(csv_file, fieldnames=["id", "name", "aisle", "department", "price"])
#    writer.writeheader()
#    for product in products:
#        writer.writerow(product)
