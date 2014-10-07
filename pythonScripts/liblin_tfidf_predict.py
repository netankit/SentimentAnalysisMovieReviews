from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
import numpy
import time
import cPickle

start = time.time()

vectorizer = TfidfVectorizer(stop_words='english',min_df=1,dtype=numpy.float64)

corpus = []
target = []
testdataset = []
phraseid= []
sentiment = []

#Training Data
with open('/mounts/Users/student/ankit/downloads/sentiment/in.txt') as f:
    for line in f:
        component = line.split('\t')
        corpus.append(component[2].rstrip('\n'))
        target.append(component[3].rstrip('\n'))

#Test Data        
with open('/mounts/Users/student/ankit/downloads/sentiment/st.tsv') as f_test:
    for line_test in f_test:
        component_test = line_test.split('\t')
	phraseid.append(component_test[0].rstrip('\n'))
        testdataset.append(component_test[2].rstrip('\n'))

# InputData FitTransform and Normalized
InputData_raw = vectorizer.fit_transform(corpus)
InputData = preprocessing.normalize(InputData_raw,norm='l2')

# TestData Fit Transform and Normalized
TestData_raw = vectorizer.transform(testdataset)
TestData = preprocessing.normalize(TestData_raw,norm='l2')

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
file = open("/mounts/Users/student/ankit/downloads/sentiment/sentiment_final.txt", "w")
file.write('PhraseId,Sentiment\n')        
for i,j in zip(phraseid , sentiment):
    val = '{0},{1}\n'.format(i, j)
    file.write(val)
file.close()

end = time.time()
print "Total execution time in minutes :: >>"
print (end - start)/60
print 'Finished!'
