'''Implement kfold for testing and training dataset'''

'''Choose features on the bases of information gain'''

'''
Take 1/5 of data as testing and 4/5 of data as training. For each training and testing pair, find decision tree, validate
with testing data, calculate accuracy and for each training and testing pair, average it.
'''
import os
import math
import random
from Assignment1.DecisionTree import DecisionTree
from Assignment1.LinearRegression import LinearRegression


class SpamBase:
    data_rows = []
    feature_values = {}
    spambase_file = "spambase.data"
    data_set = []
    k = 5

    def __init__(self):
        self.rel_path = os.path.dirname(os.path.abspath(__file__))

    '''
    Reads dataset from file and creates initial values for feature set data. 
    '''
    def load_data_set(self):
        with open(self.rel_path + '/' + self.spambase_file, 'r', encoding="utf-8") as self.spambase_file:
            self.data_rows = self.spambase_file.read().splitlines()
            for entry in self.data_rows:
                point = entry.split(',')
                point = list(map(float, point))
                self.data_set.append(point)
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
        label = feature_table[len(feature_table)-1]
        feature_table.pop(len(feature_table)-1)

        return feature_table, label

    '''
    Normalizes data through shift and scale technique. Subtracts minimum value from each and divides new maximum value
    from the new list.
    '''
    def normalize(self):
        for j in range(len(self.data_set[0])-1):
            feature_data = []
            for i in range(len(self.data_set)):
                feature_data.append(self.data_set[i][j])
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
            for i in range(len(self.data_set)):
                self.data_set[i][j] = feature_data[i]

    '''
    Takes data_set and splits training and testing data using k-fold technique. For each split, gets the classifier 
    to validate with the test data. At the end, returns the average accuracy score. 
    '''
    def train(self):
        decision_tree = DecisionTree(False)
        accuracy_score = 0
        train_data, test_data = self.kfold_split(self.k)
        for i in range(1):
            #feature_set, labels = self.generate_feature_label(train_data[i])
            classifier = decision_tree.train(train_data[i])
            accuracy_score += decision_tree.test(classifier, test_data)
            print(accuracy_score)
        # return accuracy_score / self.k
        return accuracy_score/1

    def linear_regression_train(self):
        linear_regression = LinearRegression()
        train_data, test_data = self.kfold_split(self.k)
        length = len(self.data_set[0])
        accuracy = 0
        for i in range(self.k):
            train_labels = []
            test_labels = []
            features = []
            test = []
            for val in train_data[i]:
                train_labels.append(val[length-1:])
                features.append(val[:length-1])
            for d in test_data[i]:
                test_labels.append(d[length-1:])
                test.append(d[:length-1])
            prediction_result = linear_regression.classify(features, test, train_labels)
            accuracy = accuracy + linear_regression.test(prediction_result, test_labels)
        print(accuracy/self.k)

    '''
    Partitions training and testing sets using k-fold technique.
    '''
    def kfold_split(self, k):
        data_set_size = len(self.data_set)
        test_size = math.ceil(data_set_size / k)
        train_data = []
        test_data = []
        i = 0
        while i < data_set_size:
            test_list = self.data_set[i:i + test_size]
            train_list = []
            if len(self.data_set[0:i]) != 0:
                train_list.extend(self.data_set[0:i])
            if len(self.data_set[i + test_size:data_set_size - 1]) != 0:
                train_list.extend(self.data_set[i + test_size:data_set_size - 1])
            test_data.append(test_list)
            train_data.append(train_list)
            i = i + test_size
        return train_data, test_data


def main():
    spamBase = SpamBase()
    spamBase.load_data_set()
    #spamBase.normalize()
    accuracy = spamBase.train()
    #spamBase.linear_regression_train()
    # print(accuracy)


if __name__ == '__main__':
    main()
