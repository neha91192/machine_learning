import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class NaiveBayes:

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

        # Separate X,Y
        np_X_arr = np.array(X)
        X = np.delete(np_X_arr, 57, 1)

        np_X0_arr = np.array(X0)
        X0 = np.delete(np_X0_arr, 57, 1)

        np_X1_arr = np.array(X1)
        X1 = np.delete(np_X1_arr, 57, 1)

        mean_arr = np.mean(X, axis=0)

        for i in range(len(X1)):
            for j in range(len(X1[0])):
                X1[i][j] = 0.0 if X1[i][j] < mean_arr[j] else 1.0

        for i in range(len(X0)):
            for j in range(len(X0[0])):
                X0[i][j] = 0.0 if X0[i][j] < mean_arr[j] else 1.0

        classifier = []
        classifier.append(p_spam)
        classifier.append(p_non_spam)
        classifier.append(mean_arr)
        classifier.append(X0)
        classifier.append(X1)

        return classifier

    # np.sum(
    def test(self, classifier, X, plot):
        p_spam = classifier[0]
        p_non_spam = classifier[1]
        mean_arr = classifier[2]
        bin_X0 = classifier[3]
        bin_X1 = classifier[4]

        labels = []
        for j in range(len(X)):
            labels.append(X[j][57])

        np_X_arr = np.array(X)
        X = np.delete(np_X_arr, 57, 1)
        for i in range(len(X)):
            for j in range(len(X[0])):
                X[i][j] = 0.0 if X[i][j] < mean_arr[j] else 1.0

        greater_than_mean_non_spam = np.sum(bin_X0, axis=0)
        greater_than_mean_spam = np.sum(bin_X1, axis=0)

        gtm_spam_len = len(bin_X1)
        gtm_non_spam_len = len(bin_X0)
        predictions = []
        for i in range(len(X)):
            px_y_spam = []
            px_y_non_spam = []
            for j in range(len(X[0])):
                if X[i][j] == 0.0:
                    px_y_non_spam.append((gtm_non_spam_len - greater_than_mean_non_spam[j]) / gtm_non_spam_len)
                    px_y_spam.append((gtm_spam_len - greater_than_mean_spam[j]) / gtm_spam_len)
                else:
                    px_y_non_spam.append(greater_than_mean_non_spam[j] / gtm_non_spam_len)
                    px_y_spam.append((greater_than_mean_spam[j]) / gtm_spam_len)
            pdt_spam = np.product(px_y_spam)
            pdt_non_spam = np.product(px_y_non_spam)
            if pdt_non_spam * p_non_spam > pdt_spam * p_spam:
                predictions.append(0.0)
            else:
                predictions.append(1.0)

        accuracy = self.find_accuracy(predictions, labels)
        print(confusion_matrix(labels, predictions))
        if plot:
            self.plot_graph(predictions, labels)
        return accuracy

    def find_accuracy(self, predictions, labels):
        count = 0
        for i in range(len(labels)):
            if labels[i] == predictions[i]:
                count = count + 1
        return count / len(labels)

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