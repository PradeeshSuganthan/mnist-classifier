import keras_neural_network as k
import keras_cnn as kc
import tensorflow_neural_network as t
import tensorflow_cnn as tc

accuracy = []

for i in range(5):
	accuracy.append(kc.main())

total_accuracy = sum(accuracy)/float(len(accuracy))

for i in range(len(accuracy)):
	print "Accuracy " + str(i+1) + " : " + str(accuracy[i]*100) + "%"
print "Average accuracy: " + str(total_accuracy*100) + "%"