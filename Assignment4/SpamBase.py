'''Implement kfold for testing and training dataset'''

'''Choose features on the bases of information gain'''

'''
Take 1/5 of data as testing and 4/5 of data as training. For each training and testing pair, find decision tree, validate
with testing data, calculate accuracy and for each training and testing pair, average it.
'''
import os
import math
import random
from GDA import GDA
from NaiveBayes import NaiveBayes
from NaiveBayesBin import NaiveBayesBin
from NaiveBayes9Bin import NaiveBayes9Bin
from NaiveBayes2 import NaiveBayes2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class SpamBase:
    data_rows = []
    feature_values = {}
    spambase_file = "spambase.data"
    data_set = []
    k = 10

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

    '''
    Takes data_set and splits training and testing data using k-fold technique. For each split, gets the classifier 
    to validate with the test data. At the end, returns the average accuracy score. 
    '''
    def train(self):
        gda = GDA()
        accuracy_score = 0
        train_data, test_data = self.kfold_split(self.k)
        for i in range(self.k):
            classifier = gda.train(train_data[i])
            accuracy_score = accuracy_score + gda.test(classifier, test_data[i])
        return accuracy_score/self.k


    def naive_bayes(self):
        nb = NaiveBayes()
        accuracy_score = 0
        plot = False
        train_data, test_data = self.kfold_split(self.k)
        for i in range(self.k):
            classifier = nb.train(train_data[i])
            if i == 9:
                plot = True
            accuracy_score = accuracy_score + nb.test(classifier, test_data[i], plot)
        return accuracy_score/self.k

    def naive_bayes_bin(self):
        nb = NaiveBayesBin()
        accuracy_score = 0
        plot = False
        train_data, test_data = self.kfold_split(self.k)
        for i in range(self.k):
            classifier = nb.train(train_data[i])
            if i == 9:
                plot = True
            accuracy_score = accuracy_score + nb.test(classifier, test_data[i], plot)
        return accuracy_score/self.k

    def naive_bayes_gaussian(self):
        nb = NaiveBayes2()
        accuracy_score = 0
        plot = False
        train_data, test_data = self.kfold_split(self.k)
        for i in range(self.k):
            classifier = nb.train(train_data[i])
            if i == 9:
                plot = True
            accuracy_score = accuracy_score + nb.test(classifier, test_data[i], plot)
        return accuracy_score/self.k


    def naive_bayes_9bin(self):
        nb = NaiveBayes9Bin()
        accuracy_score = 0
        plot = False
        train_data, test_data = self.kfold_split(self.k)
        for i in range(self.k):
            classifier = nb.train(train_data[i])
            if i==9:
                plot = True
            accuracy_score = accuracy_score + nb.test(classifier, test_data[i],plot)
        return accuracy_score/self.k

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
    # accuracy = spamBase.train()
    # spamBase.naive_bayes()
    # spamBase.naive_bayes_9bin()
    # spamBase.naive_bayes_bin()
    spamBase.naive_bayes_gaussian()
    # print(accuracy)




if __name__ == '__main__':
    main()
