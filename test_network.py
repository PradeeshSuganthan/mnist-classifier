import keras_perceptron as k
import keras_cnn as kc
import tensorflow_perceptron as t
import tensorflow_cnn as tc

accuracy = []

for i in range(5):
	print "Test #" + str(i + 1)
	accuracy.append(k.main())

total_accuracy = sum(accuracy)/float(len(accuracy))

for i in range(len(accuracy)):
	print "Accuracy " + str(i+1) + " : " + str(accuracy[i]*100) + "%"
print "Average accuracy: " + str(total_accuracy*100) + "%"