import os
from Assignment1.DecisionTree import DecisionTree
from Assignment1.LinearRegression import LinearRegression
import random

class Housing:
    data_rows = []
    feature_values = {}
    housing_train = "housing_train.txt"
    housing_test = "housing_test.txt"
    data_set = []
    test_set=[]

    def __init__(self):
        self.rel_path = os.path.dirname(os.path.abspath(__file__))

    '''
    Reads dataset from file and creates initial values for feature set data. 
    '''

    def load_data_set(self):
        with open(self.rel_path + '/' + self.housing_train, 'r', encoding="utf-8") as self.housing_train:
            self.data_rows = self.housing_train.read().splitlines()
            for entry in self.data_rows:
                point = entry.split()
                point = list(map(float, point))
                self.data_set.append(point)
                random.shuffle(self.data_set)


        with open(self.rel_path + '/' + self.housing_test, 'r', encoding="utf-8") as self.housing_test:
            self.data_rows = self.housing_test.read().splitlines()
            for entry in self.data_rows:
                point = entry.split()
                point = list(map(float, point))
                self.test_set.append(point)
                random.shuffle(self.data_set)

    def generate_feature_label(self, data):
        feature_table = {}
        for entry in data:
            for i, feature in enumerate(entry):
                if i in feature_table:
                    values = feature_table.get(i)
                else:
                    values = []
                values.append(feature)
                feature_table[i] = values
        label = feature_table[len(feature_table) - 1]
        feature_table.pop(len(feature_table) - 1)

        return feature_table, label

    '''
    Normalizes data through shift and scale technique. Subtracts minimum value from each and divides new maximum value
    from the new list.
    '''

    def normalize(self, data):
        for j in range(len(data[0]) - 1):
            feature_data = []
            for i in range(len(data)):
                feature_data.append(data[i][j])
            min_value = feature_data[0]
            for val in feature_data:
                min_value = min(val, min_value)
            for idx, val in enumerate(feature_data):
                feature_data[idx] = feature_data[idx] - min_value
            max_value = 0
            for val in feature_data:
                max_value = max(max_value, val)
            for idx, val in enumerate(feature_data):
                feature_data[idx] = feature_data[idx] / max_value
            for i in range(len(data)):
                data[i][j] = feature_data[i]
        return data

    '''
    Takes data_set and splits training and testing data using k-fold technique. For each split, gets the classifier 
    to validate with the test data. At the end, returns the average accuracy score. 
    '''

    def train(self):
        decision_tree = DecisionTree(True)
        classifier = decision_tree.train(self.data_set)
        accuracy_score = decision_tree.test(classifier, self.test_set)
        return accuracy_score

    def linear_regression_train(self):
        linear_regression = LinearRegression()
        length = len(self.data_set[0])
        train_labels = []
        test_labels = []
        features = []
        test = []
        for val in self.data_set:
            train_labels.append(val[length-1:])
            features.append([1]+val[:length-1])
        for d in self.test_set:
            test_labels.append(d[length-1:])
            test.append([1]+d[:length-1])
        prediction_result = linear_regression.classify(features, test, train_labels)
        accuracy = linear_regression.test(prediction_result, test_labels)
        print(accuracy)

    def linear_regression_descent(self):
        linear_regression = LinearRegression()
        length = len(self.data_set[0])
        train_labels = []
        test_labels = []
        features = []
        test = []
        for val in self.data_set:
            train_labels.append(val[length-1:])
            features.append([1]+val[:length-1])
        for d in self.test_set:
            test_labels.append(d[length-1:])
            test.append([1]+d[:length-1])
        prediction_result = linear_regression.gradient_descent(features, train_labels, test)
        print("GD")
        mse = linear_regression.test(prediction_result, test_labels)
        accuracy =  linear_regression.find_accuracy(prediction_result, test_labels, False)

        print(mse)

    def linear_regression_logistic(self):
        linear_regression = LinearRegression()
        length = len(self.data_set[0])
        train_labels = []
        test_labels = []
        features = []
        test = []
        for val in self.data_set:
            train_labels.append(val[length - 1:])
            features.append([1] + val[:length - 1])
        for d in self.test_set:
            test_labels.append(d[length - 1:])
            test.append([1] + d[:length - 1])
        prediction_result = linear_regression.logistic(features, train_labels, test)
        print("Logistic")
        mse = linear_regression.test(prediction_result, test_labels)
        #accuracy = linear_regression.find_accuracy(prediction_result, test_labels, False)

        print(mse)

    def linear_regression_ridge(self):
        linear_regression = LinearRegression()
        length = len(self.data_set[0])
        train_labels = []
        test_labels = []
        features = []
        test = []
        for val in self.data_set:
            train_labels.append(val[length - 1:])
            features.append([1] + val[:length - 1])
        for d in self.test_set:
            test_labels.append(d[length - 1:])
            test.append([1] + d[:length - 1])
        prediction_result = linear_regression.ridge(features, train_labels, test, test_labels)
        print("Ridge")
        mse = linear_regression.test(prediction_result, test_labels)
        #accuracy = linear_regression.find_accuracy(prediction_result, test_labels, False)

        print(mse)


def main():
    housing = Housing()
    housing.load_data_set()
    housing_data = housing.data_set
    data_l=len(housing.data_set)
    housing_data.extend(housing.test_set)
    data = housing.normalize(housing_data)
    housing.data_set = data[:data_l]
    housing.test_set = data[data_l:]
    # housing.data_set = housing.normalize(housing_data)
    #housing.test_set = housing.normalize(housing.test_set)
    # accuracy = housing.train()
    # print(accuracy)
    #housing.linear_regression_descent()
    housing.linear_regression_logistic()
    #housing.linear_regression_ridge()




if __name__ == '__main__':
    main()