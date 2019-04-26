from neuralnetwork import NeuralNetwork
from readdata import read_data
from functions import *


dataset = read_data()

inputs_num = len(dataset[0]) - 1
hidden_layer = 1
class_labels = get_class_labels(dataset)
outputs_num = len(class_labels)

l_rate = 0.5
epoch = 10000

network = NeuralNetwork(inputs_num, hidden_layer, outputs_num)
training(l_rate, epoch, dataset, network.network, class_labels)

#network.show_weights()