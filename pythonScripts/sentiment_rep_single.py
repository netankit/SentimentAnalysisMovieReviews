from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
import numpy
import time
import sys
import cPickle as pickle

if len(sys.argv)!=4:
        print 'Usage: python sentiment_rep_single.py <train_data> <test_data> <output_file>'
        sys.exit()


traindatapath = sys.argv[1]
testdatapath= sys.argv[2]
outputfilepath = sys.argv[3]


start = time.time()


corpus = []
target = []
testdataset = []
phraseid= []
sentiment = []

#Training Data
with open('/mounts/Users/student/ankit/downloads/sentiment/resources/in.txt') as f:
    for line in f:
        component = line.split('\t')
        corpus.append(component[2].rstrip('\n'))
        target.append(component[3].rstrip('\n'))
f.close()

#Test Data
with open('/mounts/Users/student/ankit/downloads/sentiment/resources/st.tsv') as f_test:
    for line_test in f_test:
        component_test = line_test.split('\t')
        phraseid.append(component_test[0].rstrip('\n'))
        testdataset.append(component_test[2].rstrip('\n'))
f_test.close()

# InputData FitTransform and Normalized
with open(traindatapath) as infile:
   InputData_raw = pickle.load(infile)
infile.close()
InputData = preprocessing.normalize(InputData_raw,norm='l2')


# TestData FitTransform and Normalized
with open(testdatapath, 'rb') as stfile:
   TestData_raw = pickle.load(stfile)
stfile.close()
TestData = preprocessing.normalize(TestData_raw,norm='l2')


print ('Shape of Input Data: ', InputData.shape)
print('Shape of Test Data: ',TestData.shape)

print "Starting SVM Classification "
clf_model = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
random_state=None, tol=0.0001, verbose=0)

#print "Predicting the Sentiment over Test Data"
clf = clf_model.fit(InputData,target)
out = clf.predict(TestData)
sentiment = out.tolist()

#Final Output file written to disk
file = open(outputfilepath, "w")
file.write('PhraseId,Sentiment\n')
for i,j in zip(phraseid , sentiment):
    val = '{0},{1}\n'.format(i, j)
    file.write(val)
file.close()

end = time.time()
print "Total execution time in minutes :: >>"
print (end - start)/60

startCrossValidationTime = time.time();
print"\n Predict Scores"
scores = cross_validation.cross_val_score(clf_model, InputData, target, cv=10)
print "\nFinal Accuracy Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
endCrossValidationTime = time.time();
print "Total execution time for Cross Validation in minutes :: >>"
print (endCrossValidationTime - startCrossValidationTime)/60
print 'Finished!'

