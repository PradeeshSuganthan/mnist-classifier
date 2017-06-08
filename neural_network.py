#based off of neuralnetworksanddeeplearning.com
import numpy as np 
import gzip
import struct
import random

epochs = 10 #number of training cycles
y_train = np.zeros((60000,10)) #initialize for one-hot encoding
alpha = 100 #learning rate
batchsize = 6
num_neurons

def main():
	print "Test"
	#read in data
	images_train, images_test, labels_train, labels_test = readData()

	#randomly initialize weights and biases
	weights = .01*np.random.rand(784,10)
	bias = .01*np.random.rand(10000,10)

	#one-hot encode labels
	y_train[np.arange(60000), labels_train] = 1

	#group training data
	training_data = zip(images_train, labels_train)

	#train classifier
	weights_t, bias_t = trainClassifier(epochs, images_train_b, y_train_b, weights, bias)
	#test classifier
	accuracy = testClassifier(images_test, labels_test, weights_t, bias_t)

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


def forwardPass(weights, x, bias):
	y_pred = []
	#linear model
	y_i = x.dot(weights) + bias

	#activation function
	for i in range(len(y_i)):
		y_probs = softmax(y_i[i])
		y_pred.append(y_probs)

	return y_pred

# def softmax(y):
# 	y_s = np.exp(y-np.max(y))
# 	y_soft = y_s/y_s.sum()

# 	return y_soft


def loss(y_pred, y_actual):
	#cross entropy loss
	#y_actual multiplied by log of y_pred
	#error_sum = y_actual * np.log10(y_pred-y_actual)
	#sum
	#error = -np.sum(error_sum)

	#Least squares error 
	error = np.sum((y_pred-y_actual)**2)
	return error



def sgd(training_data, weights, biases):
	#train using stochastic gradient descent
	for i in range(0,epochs):
		#randomly shuffle data
		random.shuffle(training_data)
		#partition into batches
		batches = np.split(training_data, batchsize)
		#apply gradient descent for each batch
		for batch in batches:
			weights, biases = gradientUpdate(weights, biases)

		print "Epoch " + str(i) + " complete"

	return weights, biases



def gradientUpdate(weights, bias):
	nabla_b = [np.zeros(b.shape) for b in bias]
	nabla_w = [np.zeros(w.shape) for w in weights]
	#obtain gradients
	deltaW, deltaB = backprop()

	deltaW = deltaW + nabla_w
	deltaB = deltaB + nabla_b
	#update weights & biases
	w = (weights - (alpha/len(miniBatch))*deltaw)
	b = (bias - (alpha/len(minibatch))*deltaB)

	return w, b


def backprop(x, y, weights, bias):

	nabla_b = [np.zeros(b.shape) for b in bias]
	nabla_w = [np.zeros(w.shape) for w in weights]

	#feedforward
	activation = x

	activation_list = [x]
	z_list = []

	for w, b in zip(weights, bias):
		z = np.dot(w, activation) + b
		z_list.append(z)

		activation = softmax(z)
		activation_list.append(activation)

	#backward pass
	delta = cost_derivative(activation_list[-1], y) * sigmoid_deriv(z_list[-1])

	nabla_b[-1] = delta
	nabla_w[-1] = np.dot(delta, activation_list[-2].T)

	for l in xrange(2, num_neurons):
		z = z_list[-1]
		sd = sigmoid_deriv(z)
		delta = np.dot(weights[-l + 1].T, delta) * sd
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activation_list[-l-1].T)

	return (nabla_w, nabla_b)


def cost_derivative(output, y):
	return (output - y)

def softmax(z):
	return 1.0/(1.0 + np.exp(-z))

def softmax_deriv(z):
	return softmax(z) * (1 - softmax(z))

def testClassifier(images, labels, weights, bias):
	correct = 0
	total = 0
	prediction = []

	print "Testing"
	y_pred= forwardPass(weights, images, bias)

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