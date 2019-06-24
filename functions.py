import math
from random import randrange


def neu_input(weights, inputs):
    """
    # Assume bias is the last weight in list
    total_input = 0.0
    bias = weights[-1]
    total_input += bias
    """
    # Shortcut
    total_input = weights[-1]
    for i in range(len(weights)-1):
        total_input += float(weights[i]) * float(inputs[i])
    return total_input

# 一つの入力パターンを出力層までに
# data_row : （txt の各行）入力データ
# network  : 設定したネットワーク
def feed_forward(data_row, network):
    inputs = data_row
    # 最後の層まで、各層に対して処理を繰り返す
    for layer in network:
        new_inputs = []

        # 一つの層において各ニューロンに対して処理を行う
        for neuron in layer:
            # 入力の加算
            activation = neu_input(neuron['weights'], inputs)
            neuron['output'] = neu_output(activation)
            new_inputs.append(neuron['output'])

        inputs = new_inputs
    return inputs

def neu_output(total_input):
    return sigmoid(total_input)

def sigmoid(x):
    epsilon = 1.0
    return 1.0 / (1.0 + math.exp(-(epsilon * x)))

# f'(x) = eps * (1 - f(x)) * f(x)
# f(x) = output
def sigmoid_grad(output):
    epsilon = 1.0
    return epsilon * (1.0 - output) * output

def training(l_rate, epoch, train, test, network, class_labels):
    print("============> Start training network: ")
    lowest_error = 10000
    for each in range(epoch):
        print("Epoch:", each)
        sum_error = 0
        for row_num, row in enumerate(train):
            output = feed_forward(row, network)
            target = [0.1 for i in range(len(class_labels))]
            target[class_labels[row[-1]]] = 0.9
            sum_error += error(target, output)
            back_bropagate(target, network)
            update_weights(l_rate, row, network)

        acc = predict(train, network, class_labels)
        print('train accuracy:', acc, ', loss:', sum_error)

        # For CF / 忘却時のため
        if sum_error < lowest_error:
            lowest_error = sum_error
            best_network = best_weight(network)
            #print("New lowest MSE:", lowest_error)

    final_result(train, test, network, best_network, sum_error, lowest_error, class_labels)

def back_bropagate(target, network):
    for i in reversed(range(len(network))):
        # Start from last layer
        layer = network[i]
        
        for num, neuron in enumerate(layer):
            if i == len(network) - 1:
                neuron['delta'] = sigmoid_grad(neuron['output']) * (target[num] - neuron['output'])
            else:
                sum = 0
                for later_neuron in network[i+1]:
                    sum += later_neuron['delta'] * later_neuron['weights'][num]
                neuron['delta'] = sigmoid_grad(neuron['output']) * sum

def update_weights(l_rate, row, network):
    for layer_num, layer in enumerate(network):
        # 一つの層において各ニューロン（重み）に対して処理を行う
        input_from_front_layer = []
        if layer_num == 0:
            input_from_front_layer = [float(i) for i in row[:-1]]
        else:
            input_from_front_layer = [neuron['output'] for neuron in network[layer_num - 1]]

        for neuron in layer:
            for each_i in range(len(input_from_front_layer)):
                neuron['weights'][each_i] += l_rate * neuron['delta'] * input_from_front_layer[each_i]
            neuron['weights'][-1] += l_rate * neuron['delta']

# 平均二乗誤差 (MSE) の計算
def error(targets, outputs):
    error = 0
    for i in range(len(targets)):
        error += (targets[i] - outputs[i]) ** 2
    return error / 2

# データセットのクラスラベルを記録
def get_class_labels(dataset):
    class_name = set()
    class_labels = {}

    for row in dataset[:-1]:
        if row[-1] not in class_name:
            class_labels[row[-1]] = len(class_name)
            class_name.add(row[-1])
    #print(class_labels)
    return class_labels

# データセットを学習用とテスト用に分ける
def train_test_split(dataset, split=0.6):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy

def best_weight(network):
    new_network = []
    for layer in network:
        new_network.append(layer)
    return new_network

def predict(data, network, class_labels):
    correct = 0
    for row in data:
        outputs = feed_forward(row, network)
        if class_labels[row[-1]] == outputs.index(max(outputs)):
            correct += 1
    return correct / len(data)

def final_result(train, test, network, best_network, sum_error, lowest_error, class_labels):
    print("\nNum of train data: {} , test data: {}".format(len(train), len(test)))
    print("Final weight used on data:")
    print("Final MSE:", sum_error)
    acc = predict(train, network, class_labels)
    print("Final train accuracy:", acc)
    acc = predict(test, network, class_labels)
    print("Test accuracy:", acc, "\n")

    """
    # To test for Catastrophic Forgetting / 忘却の時に利用する
    
    print("Best weight used on data:")
    print("Lowest MSE:", lowest_error)
    acc = predict(train, best_network, class_labels)
    print("Best train accuracy:", acc)
    acc = predict(test, best_network, class_labels)
    print("Test accuracy:", acc, "\n")
    """