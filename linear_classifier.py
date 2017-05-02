import numpy as np 
import gzip
import struct


def readData():
	image_train_filename = 'data/train-images-idx3-ubyte.gz'
	label_train_filename = 'data/train-labels-idx1-ubyte.gz'
	image_test_filename = 'data/t10k-images-idx3-ubyte.gz'
	label_test_filename = 'data/t10k-labels-idx1-ubyte.gz'

	print "Opening files"

	#uncompress files and read data
	with gzip.open(image_train_filename, 'r') as f:
		magicnum, numimage, row, col = struct.unpack('>IIII',f.read(16))
		images = np.fromstring(f.read(), dtype='uint8').reshape(numimage,row, col)

	with gzip.open(label_train_filename, 'r') as f:
		magicnum, numlabel = struct.unpack('>II',f.read(8))
		labels = np.fromstring(f.read(),dtype='int8')

	with gzip.open(image_test_filename, 'r') as f:
		magicnum, numimage, row, col = struct.unpack('>IIII',f.read(16))
		images_t = np.fromstring(f.read(), dtype='uint8').reshape(numimage,row, col)

	with gzip.open(label_test_filename, 'r') as f:
		magicnum, numlabel = struct.unpack('>II',f.read(8))
		labels_t = np.fromstring(f.read(),dtype='int8')


	return images, images_t, labels, labels_t

def trainClassifier():


	print "Training"





def testClassifier():


	print "Testing"




def main():
	images_train, images_test, labels_train, labels_test = readData()







if __name__ == '__main__':
	main()