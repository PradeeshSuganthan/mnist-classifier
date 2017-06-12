#based off of example code
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist

def main():
	#load data
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	#flatten data
	x_train = x_train.reshape(x_train.shape[0], 784).astype('float32')/255
	x_test = x_test.reshape(x_test.shape[0], 784).astype('float32')/255

	#conver to binary class matrices
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)

	#create model
	model = Sequential()
	model.add(Dense(500, input_shape = (784,), activation='relu'))
	model.add(Dense(200, activation='relu'))
	model.add(Dense(10, activation='softmax'))

	#model learning
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])


	#train
	model.fit(x_train, y_train, epochs = 50, batch_size = 100, validation_split = 0.1)

	score = model.evaluate(x_test, y_test)

	print 'Test loss: ' +  str(score[0])

	print 'Test accuracy: ' + str(score[1])

	return score[1]



if __name__ == '__main__':
	main()