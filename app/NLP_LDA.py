import csv
import sklearn.feature_extraction.text as text
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

## Read csv file ###
letters = []
company = []

csv_file_path = "data\shareholders_letter.csv"
with open(csv_file_path, "r") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for row in reader:
        company.append(row[1])
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

### Relationships between each company and each topic ###
company = np.asarray(company)
num_companies = len(set(company))
doctopic_grouped = np.zeros((num_companies, n_topics))

for i, name in enumerate(sorted(set(company))):
    doctopic_grouped[i, :] = np.mean(doctopic[company == name, :], axis=0)

doctopic = doctopic_grouped

print (doctopic)

### Visualization - Heatmap ###
N, K = doctopic.shape
company_names = set(company) # get companies' name
topic_labels = ['Topic #{}'.format(k) for k in range(K)] # Numbering topics

plt.pcolor(doctopic, norm=None, cmap='Blues') # Heat map
plt.yticks(np.arange(doctopic.shape[0])+0.5, company_names) # y-axis
plt.xticks(np.arange(doctopic.shape[1])+0.5, topic_labels) # x-axis
plt.gca().invert_yaxis() # flip the y-axis
plt.xticks(rotation=45) # rotate the ticks on the x-axis 45 degrees
plt.colorbar(cmap='Blues') # add a legend
plt.tight_layout() # fix margins

plt.show() # print the heatmap
