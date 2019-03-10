import pandas as pd
import numpy as np

class NeuralNetwork:

    train_data_set = []
    test_data_set = []
    X_training_data = []
    Y_training_data = []

    X_testing_data = []
    Y_testing_data = []
    W1 = []
    w2 = []
    B1 = []
    B2 = []

    l = 0.0001
    epochs = 200
    input_size = 151
    feature_size = 13
    label_size = 3
    hidden_layer_nodes = 10

    def __init__(self):
        self.W1 = np.random.randn(self.feature_size, self.hidden_layer_nodes)
        self.W2 = np.random.randn(self.hidden_layer_nodes, self.label_size)
        self.B1 = np.zeros((1, self.hidden_layer_nodes))
        self.B2 = np.zeros((1, self.label_size))


    def load_data_set(self):
        self.train_data_set = pd.read_csv("../train_wine.csv", dtype=float)
        self.test_data_set = pd.read_csv("../test_wine.csv", dtype=float)
        self.train_d = self.train_data_set.drop('F1', axis=1).values
        self.test_d =  self.test_data_set.drop('F1', axis=1).values

        data_l = len(self.train_d)
        train = self.train_d.tolist()
        train.extend(self.test_d.tolist())
        data = self.normalize(train)
        self.X_training_data = data[:data_l]
        self.X_training_data = np.array(self.X_training_data)
        self.X_testing_data = data[data_l:]
        self.X_testing_data = np.array(self.X_testing_data)

        # self.X_training_data = self.train_data_set.drop('F1', axis=1).values
        self.Y_training_data = self.train_data_set[['F1']].values
        self.Y_training_data = [self.create_label(i[0]) for i in self.Y_training_data]
        # self.Y_training_data = np.array(self.Y_training_data)
        self.Y_testing_data = self.test_data_set[['F1']].values
        self.Y_testing_data = [self.create_label(i[0]) for i in self.Y_testing_data]



    def normalize(self, data):
        for j in range(len(data[0])):
            feature_data = []
            for i in range(len(data)):
                feature_data.append(data[i][j])
            min_value = feature_data[0]
            for val in feature_data:
                min_value = min(val,min_value)
            for idx,val in enumerate(feature_data):
                feature_data[idx] = feature_data[idx] - min_value
            max_value = 0
            for val in feature_data:
                max_value = max(max_value, val)
            for idx,val in enumerate(feature_data):
                feature_data[idx] = feature_data[idx]/max_value
            for i in range(len(data)):
                data[i][j] = feature_data[i]
        return data

    def create_label(self, val):
            if val == 1.:
                return 0
            if val == 2.:
                return 1
            if val == 3.:
                return 2

    # def create_label1(self, val):
    #     if val == 1.:
    #         return [1, 0, 0]
    #     if val == 2.:
    #         return [0, 1, 0]
    #     if val == 3.:
    #         return [0, 0, 1]

    #X = 151x13
    #Y = 151x3
    #W = 13x100
    #B = 100x1
    #HL_O = 151x100
    #W2 = 100x3
    #B = 3x1
    #O = 151x3
    def forward_propagate(self):
        func = self.X_training_data.dot(self.W1) + self.B1
        hidden_layer_output = self.relu(func)

        out = hidden_layer_output.dot(self.W2) + self.B2
        output_layer = self.softmax(out) #use softmax here

        #Backpropagation
        #error in output
        output_error = output_layer.copy()
        output_error[range(self.input_size), self.Y_training_data] -= 1
        output_delta = (hidden_layer_output.T).dot(output_error)

        hidden_error = output_error.dot(self.W2.T) * self.relu_derivative(hidden_layer_output)
        # hidden_error = output_error.dot(self.W2.T)* (output_layer * (1 - output_layer)
        hidden_delta = np.dot(self.X_training_data.T, hidden_error)

        self.W2 = np.add(self.W2, self.l*output_delta)
        self.W1 = np.add(self.W1, self.l*hidden_delta)
        self.B1 = self.B1 + np.sum(hidden_delta, axis=0, keepdims=True)
        self.B2 = self.B2 + np.sum(output_delta, axis=0, keepdims=True)


        return output_layer

    def sigmoid(self, func):
        return 1/(1+np.exp(np.multiply(-1, func)))

    def relu(self, func):
        return np.maximum(func, 0)

    def relu_derivative(self, X):
        return 1.*(X>0)

    def train(self):
        for i in range(self.epochs):
            output = self.forward_propagate()
            cost = self.cross_entropy(output, self.Y_training_data)
            accuracy = self.calculate_accuracy(output, self.Y_training_data)
            print(i)
            print(cost)
            print(accuracy)

    def cross_entropy(self, output, Y):
        # y_cap = self.softmax(output)
        output[range(self.input_size), Y] -=1
        log_likelihood = np.log(output[range(self.input_size), Y])
        loss = np.sum(log_likelihood) / self.input_size
        return loss

    def softmax(self, output):
        exps = np.exp(output)
        return exps / np.sum(exps, axis=1, keepdims=True)
        # exps = np.exp(output)
        # return exps / np.sum(exps, axis=1, keepdims=True)


    def calculate_accuracy(self, output, Y):
        result = []
        match = 0
        for val in output:
            # o = [0] * (self.label_size - 1)
            x = np.argmax(val)
            # o.insert(int(x), 1)
            result.append(x)
        for i in range(len(Y)):
            if result[i] == Y[i]:
                match = match + 1
        return match / len(Y)


def main():
    neural_network = NeuralNetwork()
    neural_network.load_data_set()
    neural_network.train()




if __name__ == '__main__':
    main()