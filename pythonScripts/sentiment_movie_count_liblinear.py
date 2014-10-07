#More complex tf-idf Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import cross_validation
#from nltk.corpus import stopwords
import numpy
import time
#import cPickle
start = time.time()
#stopWords = stopwords.words('english')
vectorizer = CountVectorizer(stop_words='english',min_df=1)
#outputfile = open('C:\Users\Ankit\Documents\Python_Workspace\sentiment_movie_reviews\output_tfidf_dataset_sparse.txt', 'w')
corpus = []
target = []
testdataset = []
#Training Data
with open('/mounts/Users/student/ankit/downloads/sentiment/in.txt') as f:
    for line in f:
        component = line.split('\t')
        corpus.append(component[2].rstrip('\n'))
        target.append(component[3].rstrip('\n'))
#Test Data        
#with open('/mounts/Users/student/ankit/downloads/sentiment/st.tsv') as f_test:
#    for line_test in f_test:
#        component_test = line_test.split('\t')
#        testdataset.append(component_test[2].rstrip('\n'))
InputData = vectorizer.fit_transform(corpus)
#TestData = vectorizer.fit_transform(testdataset)

#cPickle.dump(X,outputfile,-1)
#outputfile.close()
#analyze = vectorizer.build_analyzer()
#print vectorizer.get_feature_names()
#inputsample = InputData[:150000]
#testsample=InputData[150000:]
#labelsample=target[:150000]
print "\n----------------- End of Phase 1-------- --------------------" 
print "Starting SVM Classification "
clf = svm.LinearSVC()
print "Fitting Data"
#clf.fit(inputsample, labelsample)  
print "Prediction: Check File named - result_full_training.txt"
#out = clf.predict(testsample)
#numpy.savetxt('result_full_training.txt', out,fmt='%d', delimiter=' ', newline='\n', )
print "=======================\nOUTPUT\n"
#print out
print "\n=========================\n"
print "Output file generation complete"
print "\n------------------  End of Phase 2 --------------------------"
print"\n Predict Scores"
scores = cross_validation.cross_val_score(clf, InputData, target, cv=10)
print "\nFinal Accuracy Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
end = time.time()
print "\n ------------------- End of Phase 3 --------------------------"
print "Total execution time in minutes :: >>"
print (end - start)/60
