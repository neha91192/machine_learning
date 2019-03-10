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

    l = 0.01
    #l = 0.1
    epochs = 200
    input_size = 151
    feature_size = 13
    label_size = 3
    hidden_layer_nodes = 30

    def __init__(self):
        self.W1 = np.random.randn(self.feature_size, self.hidden_layer_nodes)
        self.W2 = np.random.randn(self.hidden_layer_nodes, self.label_size)
        self.B1 = np.zeros((1, self.hidden_layer_nodes))
        self.B2 = np.zeros((1, self.label_size))



    def load_data_set(self):
        self.train_data_set = pd.read_csv("../train_wine.csv", dtype=float)
        self.test_data_set = pd.read_csv("../test_wine.csv", dtype=float)
        self.X_training_data = self.train_data_set.drop('F1', axis=1).values
        self.Y_training_data = self.train_data_set[['F1']].values
        self.Y_training_data = np.array([self.create_label(i[0]) for i in self.Y_training_data])
        self.X_testing_data = self.test_data_set.drop('F1', axis=1).values
        self.Y_testing_data = self.test_data_set[['F1']].values
        self.Y_testing_data = np.array([self.create_label(i[0]) for i in self.Y_testing_data])

    def create_label(self, val):
        if val == 1.:
            return [1, 0, 0]
        if val == 2.:
            return [0, 1, 0]
        if val == 3.:
            return [0, 0, 1]

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
        func = np.array(func, dtype=np.float128)
        hidden_layer_output = self.sigmoid(func)

        out = hidden_layer_output.dot(self.W2) + self.B2
        out = np.array(out, dtype=np.float128)
        output_layer = self.sigmoid(out)

        # backpropagate
        output_error = np.subtract(self.Y_training_data, output_layer)
        output_derivative = output_layer * (1 - output_layer)
        output_delta = output_error * output_derivative

        hidden_error = np.matmul(output_delta, (np.transpose(self.W2)))
        hidden_derivative = hidden_layer_output * (1 - hidden_layer_output)
        hidden_delta = hidden_error * hidden_derivative

        self.W1 = np.add(self.W1, self.l * np.transpose(self.X_training_data).dot(hidden_delta))
        self.W2 = np.add(self.W2, self.l * np.transpose(hidden_layer_output).dot(output_delta))
        self.B1 = np.add(self.B1, self.l * hidden_delta)
        self.B2 = np.add(self.B2, self.l * output_delta)

        # print(output_error)
        # print(self.W2)

        return output_layer

    def sigmoid(self, func):
        return 1/(1+np.exp(-func))

    def train(self):
        for i in range(self.epochs):
            output = self.forward_propagate()
            cost = self.calculate_mse(output, self.Y_training_data)
            accuracy = self.calculate_accuracy(output, self.Y_training_data)
            print(i)
            print(cost)
            print(accuracy)
        cost = self.calculate_mse(output, self.Y_training_data)
        accuracy = self.calculate_accuracy(output, self.Y_training_data)
        print("final cost")
        print(cost)
        print("final accuracy")
        print(accuracy)

    def calculate_mse(self, output, Y):
        sum_error = np.mean(np.square(np.subtract(Y, output)))
        return sum_error

    def calculate_accuracy(self, output, Y):
        # print(output)
        # print(Y)
        result =[]
        match = 0
        for val in output:
            o = [0]*(self.label_size-1)
            x = np.argmax(val)
            o.insert(int(x),1)
            result.append(o)
        for i in range(len(Y)):
            if result[i] == Y[i].tolist():
                match = match+1
        # print(result)
        return match/len(Y)



def main():
    neural_network = NeuralNetwork()
    neural_network.load_data_set()
    neural_network.train()



if __name__ == '__main__':
    main()