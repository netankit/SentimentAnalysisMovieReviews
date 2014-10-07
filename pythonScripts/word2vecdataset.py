#Dataset Creation with Word2Vec Weights
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import cPickle

vectorizer = TfidfVectorizer(min_df=1)
outputfile = open('/mounts/Users/student/ankit/downloads/indomainDataset/word2vecnew_dataset_skip_sparse.txt', 'w')
corpus = []
weights = {}
tokens = []
#Training Data
with open('/mounts/Users/student/ankit/downloads/sentiment/resources/in.txt') as f:
    for line in f:
        component = line.split('\t')
        corpus.append(component[2].rstrip('\n'))
InputData = vectorizer.fit_transform(corpus)
#print InputData
word2vecdata = InputData

print "Shape of word2vecdata initially ::>>"
print word2vecdata.shape

vcab = vectorizer.get_feature_names()
vocab_dict = {i:v for i,v in enumerate(vcab)}
#print vocab_dict

with open('/mounts/Users/student/ankit/downloads/indomainDataset/train_w2vnew_skip_avg.txt') as f:
    for line in f:
        component = line.split('\t')
        tokens.append(unicode(component[0]))
        weights[unicode(component[0])]=component[1].rstrip('\n')
#print weights

#Test 
#print "Test ::>>" 
#print vectorizer.get_feature_names() == (tokens)
print "Size of the Vectorizer:"
print len(vcab)
print "Size of the Tokens:"
print len(tokens)

print "=======sample======="
cx = scipy.sparse.coo_matrix(InputData)    
count = 0
for doc_no,position,value in zip(cx.row, cx.col, cx.data):
#   print (doc_no,position,value)
    if vocab_dict[position] in weights:
        word2vecdata[doc_no,position] = weights[vocab_dict[position]]
	count = count + 1;
print "Number of Values Changed to Word2Vec Weights:"
print count
print "====== Final Step: Generating Dataset File ====="
#print word2vecdata
cPickle.dump(word2vecdata,outputfile,-1)
outputfile.close()
print "DONE!" 
    
        
