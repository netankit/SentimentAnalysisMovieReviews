import sys
import scipy.sparse
import numpy
import cPickle as pickle

if len(sys.argv)!=5:
        print 'Usage: python datasetgeneration.py <output_file> <dataset_file> <vector_size> <train_or_test>'
        sys.exit()


outputfilepath = sys.argv[1]
datasetfile= sys.argv[2]
vector_size = int(sys.argv[3])
marker=sys.argv[4]

# DatasetFile 
with open(datasetfile, 'rb') as infile:
   data_raw = pickle.load(infile)
infile.close()

temp2 = data_raw*vector_size

#print temp2
lengthlist=[]
corpus = []

#Training Data
if marker=='train':
        with open('/mounts/Users/student/ankit/downloads/sentiment/resources/train_raw_norm.txt') as f1:
                for line1 in f1:
                        corpus.append(line1.rstrip('\n'))
	f1.close()
#Testing Data
if marker=='test':
        with open('/mounts/Users/student/ankit/downloads/sentiment/resources/test_raw_norm.txt') as f2:
                for line2 in f2:
                        corpus.append(line2.rstrip('\n'))
	f2.close()


for line in corpus:
    component = line.split(' ')
    lengthlist.append(len(component))

rowdata=[]
tempdata=[]
coldata=[]

for rownum,colnum,value in zip(temp2.row, temp2.col, temp2.data):
    dat = lengthlist[rownum]
    rowdata.append(rownum)
    coldata.append(colnum)
    tempdata.append(value/dat)

data1 = numpy.array(tempdata)
row1=  numpy.array(rowdata)
col1 =  numpy.array(coldata)
mtx = scipy.sparse.coo_matrix((data1, (row1, col1)), shape=temp2.shape)
#print mtx

outputfile = open(outputfilepath, 'w')
pickle.dump(mtx,outputfile,-1)
outputfile.close()



