import tensorflow as tf 
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True, reshape = False)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

epochs = 50
batch_size = 100
num_batches = int(mnist.train.num_examples/batch_size)
learning_rate = 0.001

#define layer sizes
l1 = 4	#conv layer output depth
l2 = 8
l3 = 12
fc = 200	#fully connected layer

#define weights/biases
#conv layers
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, l1], stddev = 0.1))
b1 = tf.Variable(tf.zeros([l1])/10)

W2 = tf.Variable(tf.truncated_normal([5, 5, l1, l2], stddev = 0.1))
b2 = tf.Variable(tf.zeros([l2])/10)

W3 = tf.Variable(tf.truncated_normal([4, 4, l2, l3]))
b3 = tf.Variable(tf.zeros([l3])/10)

#fc layer/output
W4 = tf.Variable(tf.truncated_normal([7*7*l3, fc]))
b4 = tf.Variable(tf.zeros([fc])/10)

W5 = tf.Variable(tf.truncated_normal([fc, 10]))
b5 = tf.Variable(tf.zeros([10])/10)

def neural_network():
	#Layer 1
	stride = 1
	y1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides = [1, stride, stride, 1], padding = 'SAME') + b1)

	#Layer 2
	stride = 2
	y2 = tf.nn.relu(tf.nn.conv2d(y1, W2, strides = [1, stride, stride, 1], padding = 'SAME') + b2)

	#Layer 3
	stride = 2
	y3 = tf.nn.relu(tf.nn.conv2d(y2, W3, strides = [1, stride, stride, 1], padding = 'SAME') + b3)


	#reshape output for FC layer
	yy = tf.reshape(y3, shape = [-1, 7*7*l3])

	#FC layer
	y4 = tf.nn.relu(tf.matmul(yy, W4) + b4)

	y = tf.matmul(y4, W5) + b5

	return y


def main():
	prediction = neural_network()

	#define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y_))
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	#accuracy
	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#init
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(epochs):
			for _ in range(num_batches):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				_, a = sess.run([optimizer,acc], feed_dict = {x: batch_xs, y_: batch_ys})

			print "Epoch " + str(epoch + 1) + " out of " + str(epochs) + ": Training accuracy = " + str(a*100) + "%"




		accuracy = sess.run(acc, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})

		print "Testing accuracy: " + str(accuracy*100) + "%"

		return accuracy


if __name__ == '__main__':
	main()