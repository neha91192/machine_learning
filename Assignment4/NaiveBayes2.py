import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class NaiveBayes2:

    def train(self, X):
        X0 = []
        X1 = []
        n = len(X[0]) - 1
        labels = []
        spam_count = 0
        non_spam_count = 0
        for i in range(len(X)):
            val = []
            val.append(X[i][n])
            labels.append(val)
            if X[i][n] == 0.0:
                X0.append(X[i])
                spam_count = spam_count + 1
            else:
                X1.append(X[i])
                non_spam_count = non_spam_count + 1
        p_spam = spam_count / len(X)
        p_non_spam = non_spam_count / len(X)
        np_X_arr = np.array(X)
        X_arr = np.delete(np_X_arr, 57, 1)
        np_X0_arr = np.array(X0)
        X0 = np.delete(np_X0_arr, 57, 1)

        np_X1_arr = np.array(X1)
        X1 = np.delete(np_X1_arr, 57, 1)

        mean_spam = np.mean(X1, axis=0)
        mean_non_spam = np.mean(X0, axis=0)
        covariance_mean = np.var(X1, axis=0)
        covariance_non_mean = np.var(X0, axis=0)

        classifier = []
        classifier.append(p_spam)
        classifier.append(p_non_spam)
        classifier.append(mean_spam)
        classifier.append(mean_non_spam)
        classifier.append(covariance_mean)
        classifier.append(covariance_non_mean)

        return classifier

    def test(self, classifier, test, plot):
        labels = []
        n=57
        p_spam = classifier[0]
        p_non_spam = classifier[1]
        mean_spam = classifier[2]
        mean_non_spam = classifier[3]
        covariance_spam=classifier[4]
        covariance_non_spam = classifier[5]
        for i in range(len(test)):
            labels.append(test[i][57])
        X = np.array(test)
        X = np.delete(X, 57, 1)

        p_1 = self.posterior(X,mean_spam, covariance_spam, p_spam)
        p_0 = self.posterior(X,mean_non_spam, covariance_non_spam, p_non_spam)
        predictions = 1 * (p_1 > p_0)
        accuracy = self.find_accuracy(predictions, labels)
        print(confusion_matrix(labels, predictions))
        if plot:
            self.plot_graph(predictions, labels)
        print(accuracy)
        return accuracy

    def find_accuracy(self, predictions, labels):
        count = 0
        for i in range(len(labels)):
            if labels[i] == predictions[i]:
                count = count + 1
        return count / len(labels)

    def posterior(self,X, mean, std, prob):
        product = np.prod(self.probability(X, mean, std), axis=1)
        product = product * prob
        return product

    def probability(self, x, mean, sigma):
        return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) * (1 / (np.sqrt(2 * np.pi) * sigma))


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

