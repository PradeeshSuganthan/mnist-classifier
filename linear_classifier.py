import numpy as np 
import gzip
import struct
import random
import math


epochs = 30 #number of training cycles
n_samples=60,000


def main():

	#read in data
	images_train, images_test, labels_train, labels_test = readData()

	#randomly initialize weights
	weights = np.random.rand(784,10)


	#y = linearModel(weights, images_train)

	#accuracy = loss(labels_train, y)

	#print accuracy

	#train classifier
	#weights_t = trainClassifier(epochs, images_train, labels_train, weights)

	#test classifier
	accuracy = testClassifier(images_test, labels_test, weights)

	print "Accuracy: " + str(accuracy) + "%"



def readData():
	image_train_filename = 'data/train-images-idx3-ubyte.gz'
	label_train_filename = 'data/train-labels-idx1-ubyte.gz'
	image_test_filename = 'data/t10k-images-idx3-ubyte.gz'
	label_test_filename = 'data/t10k-labels-idx1-ubyte.gz'

	print "Opening files"

	#uncompress files and read data
	with gzip.open(image_train_filename, 'r') as f:
		magicnum, numimage, row, col = struct.unpack('>IIII', f.read(16))
		images = np.fromstring(f.read(), dtype='uint8').reshape(numimage, row * col)

	with gzip.open(label_train_filename, 'r') as f:
		magicnum, numlabel = struct.unpack('>II', f.read(8))
		labels = np.fromstring(f.read(), dtype='int8')

	with gzip.open(image_test_filename, 'r') as f:
		magicnum, numimage, row, col = struct.unpack('>IIII', f.read(16))
		images_t = np.fromstring(f.read(), dtype='uint8').reshape(numimage, row * col)

	with gzip.open(label_test_filename, 'r') as f:
		magicnum, numlabel = struct.unpack('>II', f.read(8))
		labels_t = np.fromstring(f.read(), dtype='int8')

	return images, images_t, labels, labels_t


def linearModel(weights, x):
	#linear model
	y_i = x.dot(weights)
	print y_i[1]

	#activation function
	y = softmax(y_i)
	print y[1]

	return y_i

def softmax(y):
	y_soft = 1/(1+math.exp(y))

	return y_soft


def loss(y_pred, y_actual):
	print "Calculating Loss"
	#cross entropy loss
	loss = -(1/n_samples)*np.sum(y_actual*math.log10(y_pred) + (1-y_actual)*math.log10(1-y_pred))



def gradientEval(loss, weights):
	print "Evaluating Gradient"
	



def gradientUpdate(weights, weights_grad):

	w = weights + weights_grad

	return w



def trainClassifier(epochs, x, y, weights):


	print "Training"

	for i in range(0,epochs):
		y_pred= linearModel(weights, x)
		loss = loss(y_pred, y)
		gradient = gradientEval(loss, weights)
		weights = gradientUpdate(weights, gradient)


	return weights	





def testClassifier(images, labels, weights):
	correct = 0
	total = 0
	guess = []

	print "Testing"

	#Guess a random number for each image
	for i in images:
		guess.append(random.randint(0,9))


	#Check accuracy of guess
	for i in range(0,len(guess)):
		if guess[i] == labels[i]:
			correct += 1

		total += 1

	accuracy = (correct/ float(total))*100

	return accuracy



if __name__ == '__main__':
	main()