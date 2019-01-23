import os
from Assignment1.DecisionTree import DecisionTree

class Housing:
    data_rows = []
    feature_values = {}
    housing_train = "housing_train.txt"
    housing_test = "housing_test.txt"
    data_set = []
    test_set=[]
    k = 5

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

        with open(self.rel_path + '/' + self.housing_test, 'r', encoding="utf-8") as self.housing_test:
            self.data_rows = self.housing_test.read().splitlines()
            for entry in self.data_rows:
                point = entry.split()
                point = list(map(float, point))
                self.test_set.append(point)

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


def main():
    housing = Housing()
    housing.load_data_set()
    housing.data_set = housing.normalize(housing.data_set)
    housing.test_set = housing.normalize(housing.test_set)
    accuracy = housing.train()
    print(accuracy)


if __name__ == '__main__':
    main()