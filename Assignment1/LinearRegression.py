import numpy as np
import math
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


'''
add bias
'''
class LinearRegression:
    threshold = 0.4
    is_binary = True
    epoch = 50
    l = 0.1
    lg_l = 0.01
    n = 0.00000001
    ridge_thresholds = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]


    def classify(self, features, test_data, labels):
        test = np.matrix(test_data)
        wi = self.getWeights(features,labels)
        result = test.dot(wi)
        return result

    def getWeights(self, features,labels):
        X = np.matrix(features)
        Y = np.matrix(labels)
        weights = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)
        return weights


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

    def find_accuracy(self, predictions, labels, binary):
        match = 0
        spam = 0
        p_spam = 0
        diff = 2.5
        if binary:
            binary_result = []
            actual = []
            for val in predictions:
                if val[0] < self.threshold:
                    binary_result.append(0)
                else:
                    binary_result.append(1)
            for i in range(len(labels)):
                actual.append(labels[i][0])
                if labels[i][0] == binary_result[i]:
                    match = match + 1
                if labels[i][0] == 1.0:
                    spam = spam+1
                if binary_result[i] == 1.0:
                    p_spam = p_spam+1
            accuracy = match/len(labels)
            print(confusion_matrix(actual, binary_result))
            print(accuracy)
            self.plot_graph(binary_result, actual)
            return accuracy
        else:
            for i in range(len(labels)):
                if math.fabs(labels[i][0] - predictions[i][0]) <=diff:
                    match = match+1
        return match / len(labels)

    #tpr = tp/(tp+fn), fpr = (fp/fp+tn)
    def plot_graph(self, binary_result, actual):
        fpr, tpr, thresholds = roc_curve(binary_result, actual)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('SpamBase Classifier ROC')
        plt.plot(fpr, tpr, color='blue', lw=2, label='SpamBase AUC = %0.2f)' % roc_auc)
        plt.legend(loc="lower right")
        plt.show()

    def gradient_descent(self, train_data, labels, test_data):
        m = len(labels)
        n = len(train_data[0])
        wi= np.zeros(shape=(n, 1))
        X = np.matrix(train_data)
        Y = np.matrix(labels)
        test = np.matrix(test_data)
        diff=[]
        for j in range(m):
            hx = X[j].dot(wi)
            curr = (self.l*(hx-Y[j]).dot(X[j])).T
            diff.append(wi - curr)
            wi = wi - curr
        result = test.dot(wi)
        return result

    def logistic(self, train_data, labels, test_data):
        m = len(labels)
        n = len(train_data[0])
        wi = np.zeros(shape=(n, 1))
        X = np.matrix(train_data)
        Y = np.matrix(labels)
        test = np.matrix(test_data)
        for i in range(self.epoch):
            for j in range(m):
                hx = X[j].dot(wi)
                gx = 1 / (1 + math.exp(-hx))
                curr = (self.lg_l*(Y[j]-gx)*(X[j])).T
                wi = np.add(wi, curr)
        result = test.dot(wi)
        return result

    def ridge(self, train_data, labels, test_data, test_labels):
        m = len(labels)
        n = len(train_data[0])
        total = 0
        for val in labels:
            total = total + val[0]
        t_mean = total/len(labels)
        intercept = []
        for i in range(len(labels)):
            intercept.append([t_mean])
        Wmin = np.matrix(intercept)
        X = np.matrix(train_data)
        Y = np.matrix(labels)
        test = np.matrix(test_data)
        mse = sys.maxsize
        for l in self.ridge_thresholds:
            wridge = np.linalg.inv((X.T.dot(X) + l*(np.identity(n)))).dot(X.T).dot(Y)
            predictions = test.dot(wridge)
            curr_mse = self.test(predictions, test_labels)
            if curr_mse < mse:
                mse = curr_mse
                Wmin = wridge
        return test.dot(Wmin)



    def newton(self, train_data, labels, test_data):
        m = len(labels)
        n = len(train_data[0])
        wi= np.zeros(shape=(n, 1))
        X = np.matrix(train_data)
        Y = np.matrix(labels)
        test = np.matrix(test_data)
        for i in range(8):
            s = []
            hx = X.dot(wi)
            hx = np.array(hx, dtype=np.float128)
            hx = np.matrix(hx)
            pi = 1 / (1 + np.exp(-hx))
            r = pi - Y
            gk = X.T * r
            for d in range(m):
                hx = X[d].dot(wi)
                p = 1 / (1 + math.exp(-hx))
                s.append(p * (1 - p))
            sk = np.diag(s)
            Hk = X.T.dot(sk).dot(X)
            wi = wi - np.linalg.pinv(Hk)*gk
        result = test.dot(wi)
        return result



