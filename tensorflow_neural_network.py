import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

epochs = 10
batch_size = 100
num_batches = int(mnist.train.num_examples/batch_size)
learning_rate = 0.001

#define layer sizes
l_input = 784
l1 = 256
l2 = 256
l_output = 10

def neural_network(x):
	#Layer 1
	W1 = tf.Variable(tf.random_normal([l_input, l1], stddev = 0.1))
	b1 = tf.Variable(tf.zeros([l1]))

	y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

	#Layer 2
	W2 = tf.Variable(tf.random_normal([l1, l2], stddev = 0.1))
	b2 = tf.Variable(tf.zeros([l2]))

	y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

	#Layer 3
	W3 = tf.Variable(tf.random_normal([l2, l_output]))
	b3 = tf.Variable(tf.zeros([l_output]))

	y3 = tf.matmul(y2, W3) + b3

	#y = tf.nn.softmax(y3)

	return y3



def main():
	prediction = neural_network(x)

	#define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y_))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

	#init
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(epochs):
			for _ in range(num_batches):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				_ = sess.run(optimizer, feed_dict = {x: batch_xs, y_: batch_ys})

			print "epoch " + str(epoch + 1) + " out of " + str(epochs)

		#accuracy
		correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
		acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		accuracy = sess.run(acc, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})

		print "Testing accuracy: " + str(accuracy)

		return accuracy


if __name__ == '__main__':
	main()