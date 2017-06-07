#based off of example code
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist



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
model.add(Dense(10, input_shape = (784,)))
model.add(Activation('softmax'))

#model learning
model.compile(optimizer='SGD', loss='mse', metrics = ['accuracy'])


#train
model.fit(x_train, y_train, epochs = 10, batch_size = 128, validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test)

print 'Test loss: ' +  str(score[0])

print 'Test accuracy: ' + str(score[1])