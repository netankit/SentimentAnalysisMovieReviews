#Dataset Creation with different representation weights for Training Data

import sys
import scipy.sparse
import numpy
import cPickle


if len(sys.argv)!=6:
        print 'Usage: python datasetgeneration.py <output_file> <raw_vector_file> <vector_size> <vocabulary_size> <train_or_test>'
        sys.exit()


outputfilepath = sys.argv[1]
vectorsfilepath= sys.argv[2]
vector_size = int(sys.argv[3])
sample_size=int(sys.argv[4])
marker=sys.argv[5]
corpus = []

if marker=='train':
#Training Data
	with open('/mounts/Users/student/ankit/downloads/sentiment/resources/train_raw_norm.txt') as f:
    		for line1 in f:
        		corpus.append(line1.rstrip('\n'))
if marker=='test':
	with open('/mounts/Users/student/ankit/downloads/sentiment/resources/test_raw_norm.txt') as f:
                for line2 in f:
                        corpus.append(line2.rstrip('\n'))

outputfile = open(outputfilepath, 'w')
repvector = []
vocab=[]
tokens = []
sentencevectortemp = []
sentencevector = []
corpusvector=[]

#Size of Vector {Default: W2V=100, Poly=64, Glove=50}

# Raw Vector File
with open(vectorsfilepath) as p:
	for line in p:
		if line.rstrip('\n') != str(sample_size)+' '+str(vector_size):
			component2 = line.split(' ')
			vocab.append(component2[0])
			repvector.append(map(float,component2[1:vector_size+1]))
vocab_dict = {v:i for i,v in enumerate(vocab)}

for line in corpus:
	component_word = line.split(' ')
	for one_word in component_word:
		if one_word in  vocab_dict:
			indexnum = vocab_dict.get(one_word)
			sentencevectortemp.append(repvector[indexnum])
	l = numpy.array(sentencevectortemp)
	lmt = numpy.array(l.sum(axis=0, dtype=numpy.float32)).tolist()
	lmt[:] = [x / vector_size for x in lmt]
	sentencevector.append(lmt)
t = numpy.array(sentencevector)
temp = scipy.sparse.coo_matrix(t)
cPickle.dump(temp,outputfile,-1)
outputfile.close()



