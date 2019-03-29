import numpy as np
import math

class GDA:

    def train(self, X):
        size = len(X)
        spam_arr = []
        non_spam_arr = []
        spam_count = 0
        non_spam_count = 0
        for i in range(len(X)):
            if X[i][57] == 0.0:
                non_spam_arr.append(X[i])
                non_spam_count = non_spam_count + 1
            else:
                spam_arr.append(X[i])
                spam_count = spam_count + 1
        np_spam_arr = np.array(spam_arr)
        sp_arr = np.delete(np_spam_arr, 57, 1)
        np_non_spam_arr = np.array(non_spam_arr)
        non_sp_arr = np.delete(np_non_spam_arr, 57, 1)

        np_X_arr = np.array(X)
        X_arr = np.delete(np_X_arr, 57, 1)
        phi_spam = spam_count/size
        phi_non_spam = non_spam_count/size
        mean_spam = np.mean(sp_arr, axis=0)
        mean_non_spam = np.mean(non_sp_arr, axis=0)
        covariance = np.cov(X_arr.T)
        result = []
        result.append(phi_spam)
        result.append(phi_non_spam)
        result.append(mean_spam)
        result.append(mean_non_spam)
        result.append(covariance)
        return result


    def test(self, classifier, test):
        labels = []
        n=57
        mean_spam = classifier[2]
        mean_non_spam = classifier[3]
        covariance=classifier[4]
        for i in range(len(test)):
            labels.append(test[i][57])
        X = np.array(test)
        X_arr = np.delete(X, 57, 1)
        predictions = []
        for x in X_arr:
            a= math.pow(2*math.pi,n/2)
            b= math.pow(np.linalg.det(covariance),1/2)
            left = 1/(a*b)
            p1 = x - mean_spam
            p2 = x - mean_non_spam
            q = np.linalg.pinv(covariance)
            p_spam = left*np.exp(-0.5*np.dot(np.dot(p1.T,q),p1))
            p_non_spam = left*np.exp(-0.5*np.dot(np.dot(p2.T,q),p2))
            predictions.append(0.0 if p_non_spam > p_spam else 1.0)
        accuracy = self.find_accuracy(predictions, labels)
        print(accuracy)
        return accuracy


    def find_accuracy(self, predictions, labels):
        count = 0
        for i in range(len(labels)):
            if labels[i]==predictions[i]:
                count= count+1
        return count/len(labels)