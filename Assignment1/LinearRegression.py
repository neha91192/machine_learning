import numpy as np
import math

'''
add bias
'''
class LinearRegression:
    threshold = 0.44
    is_binary = False
    iterations = 1000
    l = 0.00001

    def classify(self, features, test_data, labels):
        X = np.matrix(features)
        Y = np.matrix(labels)
        test = np.matrix(test_data)
        weights = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)
        result = test.dot(weights)
        return result

    def test(self, predictions, labels):
        square = 0
        if self.is_binary:
            binary_result = []
            for val in predictions:
                if val[0] < self.threshold:
                    binary_result.append(0)
                else:
                    binary_result.append(1)
            for i in range(len(labels)):
                diff = labels[i][0] - binary_result[i]
                square = square + math.pow(diff, 2)
        else:
            for i in range(len(labels)):
                diff = labels[i][0] - predictions[i][0]
                square = square + math.pow(diff, 2)
        return square / len(labels)

    def gradient_descent(self, train_data, labels, test_data):
        m = len(labels)
        n = len(train_data[0])
        wi = np.random.randn(n,1)
        X = np.matrix(train_data)
        Y = np.matrix(labels)

        for i in range(self.iterations):
            hx = X.T.dot(wi)
            wi = wi - self.l*(2/m)*(Y.T-hx).dot(X)
        result = test_data.dot(wi)

        return result

