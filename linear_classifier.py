import numpy as np 
import gzip
import struct
import random

epochs = 30 #number of training cycles
y_train = np.zeros((60000,10)) #initialize for one-hot encoding
alpha = 100 #learning rate
batchsize = 6


def main():
	print "Test"
	#read in data
	images_train, images_test, labels_train, labels_test = readData()

	#randomly initialize weights
	weights = .01*np.random.rand(784,10)

	#one-hot encode labels
	y_train[np.arange(60000), labels_train] = 1

	print "Pre-batching"
	#batch data
	images_train_b = np.split(images_train, batchsize)

	y_train_b = np.split(y_train, batchsize)

	#train classifier
	weights_t = trainClassifier(epochs, images_train_b, y_train_b, weights)
	#test classifier
	accuracy = testClassifier(images_test, labels_test, weights_t)

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
	y_pred = []
	#linear model
	y_i = x.dot(weights)

	#activation function
	for i in range(len(y_i)):
		y_probs = softmax(y_i[i])
		y_pred.append(y_probs)

	return y_pred

def softmax(y):
	y_s = np.exp(y-np.max(y))
	y_soft = y_s/y_s.sum()

	return y_soft


def loss(y_pred, y_actual):
	#cross entropy loss
	#y_actual multiplied by log of y_pred
	#error_sum = y_actual * np.log10(y_pred)
	#sum
	#error = -np.sum(error_sum)

	#Least squares error 
	error = np.sum((y_pred-y_actual)**2)
	return error



def gradientEval(X, y_pred, y_actual):

	gradient = (y_actual - y_pred).T.dot(X)

	return gradient



def gradientUpdate(weights, weights_grad):

	w = weights + alpha * weights_grad.T

	return w



def trainClassifier(epochs, x, y, weights):


	print "Training"



	for i in range(0,batchsize):
		print "Batch #" + str(i + 1) + ": "
		for j in range(0, epochs):
			y_pred= linearModel(weights, x[i])
			cost = loss(y_pred, y[i])
			gradient = gradientEval(x[i], y_pred, y[i])
			weights = gradientUpdate(weights, gradient)

			print "Cost " + str(j) + ": " + str(cost)


	return weights	





def testClassifier(images, labels, weights):
	correct = 0
	total = 0
	prediction = []

	print "Testing"
	y_pred= linearModel(weights, images)

	#predictions for test images
	for i in range(len(y_pred)):
		prediction.append(np.argmax(y_pred[i]))

	#Check accuracy of guess
	for i in range(0,len(y_pred)):
		if prediction[i] == labels[i]:
			correct += 1

		total += 1

	accuracy = (correct/ float(total))*100

	return accuracy



if __name__ == '__main__':
	main()