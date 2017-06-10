import keras_neural_network as k
import tensorflow_neural_network as t

accuracy = []

for i in range(5):
	accuracy.append(t.main())

total_accuracy = sum(accuracy)/float(len(accuracy))

for i in range(len(accuracy)):
	print "Accuracy " + str(i+1) + " : " + str(accuracy[i]*100) + "%"
print "Average accuracy: " + str(total_accuracy*100) + "%"