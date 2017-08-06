import csv
import sklearn.feature_extraction.text as text
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

### customize function ###
def read_letter_from_file(csv_file_path):
    letters = []
    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            letters.append(row[3])
    return letters

def read_company_from_file(csv_file_path):
    company = []
    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            company.append(row[1])
    return company

def number_of_comapany(company):
    return len(set(company))

def name_of_company(company):
    return set(company)


### run the code ###
def run():
    file_path = "data\shareholders_letter.csv"
    letters = read_letter_from_file(csv_file_path = file_path)  # read letter, test 1
    company = read_company_from_file(csv_file_path = file_path)  # read company, test 2

    ### Change the texts into Documents Term Matix (dtm) ####
    # remove stopwords and words that appear less than five times
    vectorizer = text.CountVectorizer(input = letters,  stop_words = 'english', lowercase = True, min_df = 10)
    dtm = vectorizer.fit_transform(letters) # create dtm
    vocab = np.array(vectorizer.get_feature_names()) # list of words and change it to an array


    ### Generate ten topics & ten top words in each topic ###
    n_topics = 10
    n_top_words = 10
    # classifier
    lda = LatentDirichletAllocation(n_topics = n_topics, random_state=0)
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
    num_companies = number_of_comapany(company) # number of company, test 3
    doctopic_grouped = np.zeros((num_companies, n_topics))

    for i, name in enumerate(sorted(set(company))):
        doctopic_grouped[i, :] = np.mean(doctopic[company == name, :], axis=0)

    doctopic = doctopic_grouped

    print (doctopic)


    ### Visualization - Heatmap ###
    N, K = doctopic.shape
    company_names = name_of_company(company) # get companies' name, test 4
    topic_labels = ['Topic #{}'.format(k) for k in range(K)] # Numbering topics

    plt.pcolor(doctopic, norm=None, cmap='Blues') # Heat map
    plt.yticks(np.arange(doctopic.shape[0])+0.5, company_names) # y-axis
    plt.xticks(np.arange(doctopic.shape[1])+0.5, topic_labels) # x-axis
    plt.gca().invert_yaxis() # flip the y-axis
    plt.xticks(rotation=45) # rotate the ticks on the x-axis 45 degrees
    plt.colorbar(cmap='Blues') # add a legend
    plt.tight_layout() # fix margins

    plt.show() # print the heatmap


    ### Visualization - topic distance ###
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_similarity

    dist = 1 - cosine_similarity(dtm) # distance

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    pos = mds.fit_transform(dist)
    xs, ys = pos[:, 0], pos[:, 1]
    for x, y, name in zip(xs, ys, company_names):
        plt.scatter(x, y)
        plt.text(x, y, name)

    plt.show()

if __name__ == "__main__":
    run()
