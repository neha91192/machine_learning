import os
import math
import numpy as np

class Perceptron:
    file = "perceptronData"
    data_rows = []
    data_set = []
    k = 5
    l = 0.0001


    def __init__(self):
        self.rel_path = os.path.dirname(os.path.abspath(__file__))

    def load_data_set(self):
        with open(self.rel_path + '/' + self.file, 'r', encoding="utf-8") as self.file:
            self.data_rows = self.file.read().splitlines()
            for entry in self.data_rows:
                point = entry.split()
                point = list(map(float, point))
                self.data_set.append(point)
                # random.shuffle(self.data_set)

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

    def run(self):
        # train_data, test_data = self.kfold_split(self.k)
        length = len(self.data_set[0])
        # for i in range(1):
        train_labels = []
        test_labels = []
        features = []
        test = []
        for val in self.data_set:
            train_labels.append(val[length - 1:])
            features.append([1] + val[:length - 1])
        # for d in test_data[i]:
        #     test_labels.append(d[length - 1:])
        #     test.append([1] + d[:length - 1])
        prediction_result = self.perceptron_algorithm(features, train_labels)


    def perceptron_algorithm(self, train_data, labels):
        m = len(labels)
        n = len(train_data[0])
        wi = np.zeros(shape=(n, 1))
        X = np.matrix(train_data)
        Y = np.matrix(labels)
        itr = 0
        while True:
            mis = 0
            itr = itr+1
            for i in range(m):
                val = Y[i].dot(X[i].dot(wi))
                if val[0] <= 0:
                    wi = wi + self.l*(Y[i].dot(X[i])).T
                    mis = mis+1
            print("Iteration "+str(itr)+", "+"total_mistake "+str(mis))
            if mis == 0:
                break
        print("Classifier weights:")
        print(wi)
        t = wi[0][0]
        print("Normalized with thresholds")
        for i in range(1, len(wi)):
            print(-wi[i][0]/t)
        return wi



def main():
    perceptron = Perceptron()
    perceptron.load_data_set()
    perceptron.run()




if __name__ == '__main__':
    main()