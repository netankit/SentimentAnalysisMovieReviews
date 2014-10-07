from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
import numpy
import time

import cPickle as pickle

start = time.time()

vectorizer = CountVectorizer(stop_words='english',min_df=1,dtype=numpy.float64)

corpus = []
newscorpus = []
target = []
testdataset = []
phraseid= []
sentiment = []

#OutDomain Dataset
with open('/mounts/Users/student/ankit/downloads/outdomainDataset/news_norm_raw.txt') as outf:
    for line in outf:
        newscorpus.append(line.rstrip('\n'))
        
#Training Data
with open('/mounts/Users/student/ankit/downloads/sentiment/resources/in.txt') as f:
    for line in f:
        component = line.split('\t')
        corpus.append(component[2].rstrip('\n'))
        target.append(component[3].rstrip('\n'))

#Test Data
with open('/mounts/Users/student/ankit/downloads/sentiment/resources/st.tsv') as f_test:
    for line_test in f_test:
        component_test = line_test.split('\t')
        phraseid.append(component_test[0].rstrip('\n'))
        testdataset.append(component_test[2].rstrip('\n'))

# NewsData FitTransform and Normalized
NewsData_raw = vectorizer.fit_transform(newscorpus)
NewsData = preprocessing.normalize(NewsData_raw,norm='l2')

## BAG OF WORDS
# InputData FitTransform and Normalized
InputDataB_raw = vectorizer.transform(corpus)
InputDataB = preprocessing.normalize(InputDataB_raw,norm='l2')

# TestData Fit Transform and Normalized
TestDataB_raw = vectorizer.transform(testdataset)
TestDataB = preprocessing.normalize(TestDataB_raw,norm='l2')

##WORD2VEC
# InputData FitTransform and Normalized
with open('/mounts/Users/student/ankit/downloads/final_datasets/outdomain/poly_news_skip_norm.txt', 'rb') as infile:
   InputDataW_raw = pickle.load(infile)
InputDataW = preprocessing.normalize(InputDataW_raw,norm='l2')

# TestData FitTransform and Normalized
with open('/mounts/Users/student/ankit/downloads/final_datasets/outdomain/poly_news_skip_norm_test.txt', 'rb') as stfile:
   TestDataW_raw = pickle.load(stfile)
TestDataW = preprocessing.normalize(TestDataW_raw,norm='l2')

#Concatinating the Bag of Word vectors with the Word2Vec Vectors
InputData = hstack((InputDataW, InputDataB))
TestData = hstack((TestDataW, TestDataB))

print ('Shape of Input Data: ', InputData.shape)
print('Shape of Test Data: ',TestData.shape)

print "Starting SVM Classification "
clf_model = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
random_state=None, tol=0.0001, verbose=0)

print "Predicting the Sentiment over Test Data"
clf = clf_model.fit(InputData,target)
out = clf.predict(TestData)
sentiment = out.tolist()

#Final Output file written to disk
file = open("/mounts/Users/student/ankit/downloads/sentiment/result/output_outdomain_news_w2v_poly.txt", "w")
file.write('PhraseId,Sentiment\n')
for i,j in zip(phraseid , sentiment):
    val = '{0},{1}\n'.format(i, j)
    file.write(val)
file.close()

end = time.time()
print "Total execution time in minutes :: >>"
print (end - start)/60
print 'Finished!'


