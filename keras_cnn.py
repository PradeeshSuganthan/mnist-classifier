#based off of example code
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist

def main():
	#load data
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	#flatten data
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255

	#conver to binary class matrices
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)

	input_shape = (28, 28, 1)


	#create model
	model = Sequential()
	model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
	model.add(Conv2D(64, (3, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation = 'relu'))
	model.add(Dense(10, activation = 'softmax'))

	#model learning
	model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics = ['accuracy'])


	#train
	model.fit(x_train, y_train, epochs = 20, batch_size = 128, validation_data = (x_test, y_test))

	score = model.evaluate(x_test, y_test)

	print 'Test loss: ' +  str(score[0])

	print 'Test accuracy: ' + str(score[1])

	return score[1]



if __name__ == '__main__':
	main()