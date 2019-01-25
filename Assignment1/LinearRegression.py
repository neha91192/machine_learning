import numpy as np
import math


class LinearRegression:
    threshold = 0.44
    is_binary = False

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
        print(square/len(labels))
        return square / len(labels)
