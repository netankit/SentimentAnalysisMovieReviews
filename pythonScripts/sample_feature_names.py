#Dataset Creation with Word2Vec Weights
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import cPickle

vectorizer = TfidfVectorizer(min_df=1)
#outputfile = open('/mounts/Users/student/ankit/downloads/indomainDataset/word2vecnew_dataset_skip_sparse.txt', 'w')
corpus = []
weights = {}
tokens = []
#Training Data
with open('/mounts/Users/student/ankit/downloads/sentiment/resources/example.txt') as f:
    for line in f:
        component = line.split('\t')
        corpus.append(component[2].rstrip('\n'))
InputData = vectorizer.fit_transform(corpus)
#print InputData
word2vecdata = InputData

print "Shape of word2vecdata initially ::>>"
print word2vecdata.shape

vcab = vectorizer.get_feature_names()
print "vcab:\n"
print vcab
