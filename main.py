from neuralnetwork import NeuralNetwork
from readdata import read_data
from functions import *


dataset = read_data()
train, test = train_test_split(dataset, 0.7)

#print(len(train))
#print(len(test))

inputs_num = len(dataset[0]) - 1
hidden_layer = 1
class_labels = get_class_labels(dataset)
outputs_num = len(class_labels)

l_rate = 0.05
epoch = 10000

model = NeuralNetwork(inputs_num, hidden_layer, outputs_num)
#model = NeuralNetwork(inputs_num, 2, outputs_num, 10, 10)
training(l_rate, epoch, train, test, model.network, class_labels)

#model.show_weights()