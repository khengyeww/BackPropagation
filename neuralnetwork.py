from random import random


class NeuralNetwork:

    # 中間層の数の決め方？
    # One hidden layer is sufficient for the large majority of problems.
    # 中間層のニューロン数の決め方？
    # 1. The number of hidden neurons should be between the size of the input layer and the size of the output layer.
    # -> np.random.randn(input_size, hidden_size) and np.random.randn(hidden_size, output_size)
    # 2. The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
    # 3. The number of hidden neurons should be less than twice the size of the input layer.

    # 出力層のニューロン数の決め方？
    # 1. If the NN is a regressor, then the output layer has a single node.
    # 2. If the NN is a classifier, then it also has a single node unless softmax is used in which case
    #      the output layer has one node per class label in your model.
    #def __init__(self, inputs_num, *args, hidden_layer, outputs_num, hidden_neu_num = False):
    def __init__(self, inputs_num, hidden_layer, outputs_num, *args):
        self.inputs_num = inputs_num

        if hidden_layer == 0:
            hidden_layer = 1
        self.hidden_layer = hidden_layer
        self.outputs_num = outputs_num
        
        default_hidden_num = int((inputs_num * 2 / 3) + outputs_num)

        hidden_num_arr = []
        if len(args) == 0:
            [hidden_num_arr.append(default_hidden_num) for i in range(self.hidden_layer)]
        elif len(args) >= self.hidden_layer:
            [hidden_num_arr.append(int(args[i])) for i in range(self.hidden_layer)]
        else:
            remaining = self.hidden_layer - len(args)
            [hidden_num_arr.append(int(args[i])) for i in range(len(args))]
            [hidden_num_arr.append(default_hidden_num) for i in range(remaining)]

        self.hidden_num_arr = hidden_num_arr

        # 初期重みの生成
        network = []

        # 入力層と中間層の間
        input_layer = [{"weights":[random() for i in range(self.inputs_num + 1)]} for i in range(self.hidden_num_arr[0])] # Extra 1 weight as bias
        network.append(input_layer)

        # 中間層と中間層の間（中間層数 > 2 の場合のみ）
        for i in range(self.hidden_layer - 1):
            weights_per_layer = [{"weights":[random() for j in range(self.hidden_num_arr[i] + 1)]} for j in range(self.hidden_num_arr[i+1])] # Extra 1 weight as bias
            network.append(weights_per_layer)

        # 中間層と出力層の間
        output_layer = [{"weights":[random() for i in range(self.hidden_num_arr[-1] + 1)]} for i in range(self.outputs_num)] # Extra 1 weight as bias
        network.append(output_layer)

        self.network = network

    # 確認のため
    def show_weights(self):
        print()
        print('Number of neuron(s) in input layer:', self.inputs_num)
        print('Number of hidden layer(s):', self.hidden_layer)
        print('Number of neuron(s) in output layer:', self.outputs_num)
        print('***Note: Last weight as bias')
        print()
        print()
        for count, layer in enumerate(self.network):
            count += 1
            if count != len(self.network):
                print('Number of neuron(s) in hidden layer{}: {}'.format(count, len(layer)))
            else:
                print('Number of neuron(s) in output layer: {}'.format(len(layer)))
            
            print('Weights from previous layer\'s neuron(s) connected to current layer\'s:')
            for num, each_neu in enumerate(layer):
                print("     Neuron {}:".format(num+1), each_neu['weights'])
            print()