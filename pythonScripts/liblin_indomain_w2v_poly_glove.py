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
#newscorpus = []
target = []
testdataset = []
phraseid= []
sentiment = []

#OutDomain Dataset
#with open('/mounts/Users/student/ankit/downloads/word2vecnew/trunk/news.2012.en.shuffled') as outf:
#    for line in outf:
#        newscorpus.append(line.rstrip('\n'))
        
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
##Word2Vec
# InputData FitTransform and Normalized
with open('/mounts/Users/student/ankit/downloads/final_datasets/indomain/w2v_train_dataset_norm.txt', 'rb') as infile:
   InputDataW_raw = pickle.load(infile)
InputDataW = preprocessing.normalize(InputDataW_raw,norm='l2')


# TestData FitTransform and Normalized
with open('/mounts/Users/student/ankit/downloads/final_datasets/indomain/w2v_test_dataset_norm.txt', 'rb') as stfile:
   TestDataW_raw = pickle.load(stfile)
TestDataW = preprocessing.normalize(TestDataW_raw,norm='l2')

##PolyGlot
# InputData FitTransform and Normalized
with open('/mounts/Users/student/ankit/downloads/final_datasets/indomain/poly_dataset_train_skip.txt', 'rb') as infile:
   InputDataP_raw = pickle.load(infile)
InputDataP = preprocessing.normalize(InputDataP_raw,norm='l2')


# TestData FitTransform and Normalized
with open('/mounts/Users/student/ankit/downloads/final_datasets/indomain/poly_dataset_test_skip.txt', 'rb') as stfile:
   TestDataP_raw = pickle.load(stfile)
TestDataP = preprocessing.normalize(TestDataP_raw,norm='l2')


##Glove
# InputData FitTransform and Normalized
with open('/mounts/Users/student/ankit/downloads/final_datasets/indomain/glove_dataset_train_skip.txt', 'rb') as infile:
   InputDataG_raw = pickle.load(infile)
InputDataG = preprocessing.normalize(InputDataG_raw,norm='l2')


# TestData FitTransform and Normalized
with open('/mounts/Users/student/ankit/downloads/final_datasets/indomain/glove_dataset_test_skip.txt', 'rb') as stfile:
   TestDataG_raw = pickle.load(stfile)
TestDataG = preprocessing.normalize(TestDataG_raw,norm='l2')




# NewsData FitTransform and Normalized
#NewsData_raw = vectorizer.fit_transform(newscorpus)
#NewsData = preprocessing.normalize(NewsData_raw,norm='l2')

## BAG OF WORDS
# InputData FitTransform and Normalized
InputDataB_raw = vectorizer.fit_transform(corpus,)
InputDataB = preprocessing.normalize(InputDataB_raw,norm='l2')
#InputDataB =  vectorizer.fit_transform(corpus)


# TestData Fit Transform and Normalized
TestDataB_raw = vectorizer.transform(testdataset)
TestDataB = preprocessing.normalize(TestDataB_raw,norm='l2')
#TestDataB = vectorizer.transform(testdataset)

print ('Shape of Input DataB: ', InputDataB.shape)
print ('Shape of Input DataW: ', InputDataW.shape)
print('Shape of Test DataB: ',TestDataB.shape)
print('Shape of Test DataW: ',TestDataW.shape)

InputData = hstack((InputDataW, InputDataP, InputDataG))
TestData = hstack((TestDataW, TestDataP, TestDataG))

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
file = open("/mounts/Users/student/ankit/downloads/sentiment/result/output_indomain_w2v_poly_glove.txt", "w")
file.write('PhraseId,Sentiment\n')
for i,j in zip(phraseid , sentiment):
    val = '{0},{1}\n'.format(i, j)
    file.write(val)
file.close()

end = time.time()
print "Total execution time in minutes :: >>"
print (end - start)/60

print 'Finished!'


