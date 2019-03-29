import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class NaiveBayesBin:

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
        min_arr = np.amin(X, axis=0)
        max_arr = np.amax(X, axis=0)

        mean_spam_arr = np.mean(X1, axis=0)
        mean_non_spam_arr = np.mean(X0, axis=0)

        for i in range(len(X1)):
            for j in range(len(X1[0])):
                low_mean = mean_spam_arr[j] if mean_spam_arr[j] < mean_arr[j] else mean_non_spam_arr[j]
                high_mean = mean_spam_arr[j] if mean_spam_arr[j] > mean_arr[j] else mean_non_spam_arr[j]

                if X1[i][j] >= min_arr[j] and X1[i][j] < low_mean:
                    X1[i][j] = 1
                elif X1[i][j] >= low_mean and X1[i][j] < mean_arr[j]:
                    X1[i][j] = 2
                elif X1[i][j] >= mean_arr[j] and X1[i][j] < high_mean:
                    X1[i][j] = 3
                elif X1[i][j] >= high_mean and X1[i][j] <= max_arr[j]:
                    X1[i][j] = 4


        for i in range(len(X0)):
            for j in range(len(X0[0])):
                low_mean = mean_spam_arr[j] if mean_spam_arr[j] < mean_arr[j] else mean_non_spam_arr[j]
                high_mean = mean_spam_arr[j] if mean_spam_arr[j] > mean_arr[j] else mean_non_spam_arr[j]

                if X0[i][j] >= min_arr[j] and X0[i][j] < low_mean:
                    X0[i][j] = 1
                elif X0[i][j] >= low_mean and X0[i][j] < mean_arr[j]:
                    X0[i][j] = 2
                elif X0[i][j] >= mean_arr[j] and X0[i][j] < high_mean:
                    X0[i][j] = 3
                elif X0[i][j] >= high_mean and X0[i][j] <= max_arr[j]:
                    X0[i][j] = 4


        classifier = []
        classifier.append(p_spam)
        classifier.append(p_non_spam)
        classifier.append(mean_spam_arr)
        classifier.append(mean_non_spam_arr)
        classifier.append(X1)
        classifier.append(X0)
        classifier.append(min_arr)
        classifier.append(max_arr)
        classifier.append(mean_arr)

        return classifier

    def test(self, classifier, X, plot):
        p_spam = classifier[0]
        p_non_spam = classifier[1]
        mean_spam_arr = classifier[2]
        mean_non_spam_arr = classifier[3]
        bin_X1 = classifier[4]
        bin_X0 = classifier[5]
        min_arr = classifier[6]
        max_arr = classifier[7]
        mean_arr = classifier[8]

        labels = []
        for j in range(len(X)):
            labels.append(X[j][57])

        np_X_arr = np.array(X)
        X = np.delete(np_X_arr, 57, 1)
        for i in range(len(X)):
            for j in range(len(X[0])):
                low_mean = mean_spam_arr[j] if mean_spam_arr[j] < mean_arr[j] else mean_non_spam_arr[j]
                high_mean = mean_spam_arr[j] if mean_spam_arr[j] > mean_arr[j] else mean_non_spam_arr[j]

                if X[i][j] >= min_arr[j] and X[i][j] < low_mean:
                    X[i][j] = 1
                elif X[i][j] >= low_mean and X[i][j] < mean_arr[j]:
                    X[i][j] = 2
                elif X[i][j] >= mean_arr[j] and X[i][j] < high_mean:
                    X[i][j] = 3
                elif X[i][j] >= high_mean and X[i][j] <= max_arr[j]:
                    X[i][j] = 4
                else:
                    if X[i][j] < min_arr[j]:
                        X[i][j]=1
                    elif X[i][j]>max_arr[j]:
                        X[i][j]=4

        X_tran_X0 = bin_X0.T
        X_tran_X1 = bin_X1.T

        count_table_X0 = []
        for row in X_tran_X0:
            col_count = []
            col_count.append(0)
            col_count.append(0)
            col_count.append(0)
            col_count.append(0)
            for i in range(len(row)):
                col_count[int(row[i]-1)] +=1
            count_table_X0.append(col_count)

        count_table_X1 = []
        for row in X_tran_X1:
            col_count = []
            col_count.append(0)
            col_count.append(0)
            col_count.append(0)
            col_count.append(0)
            for i in range(len(row)):
                col_count[int(row[i]-1)] += 1
            count_table_X1.append(col_count)

        gtm_spam_len = len(bin_X1)
        gtm_non_spam_len = len(bin_X0)
        predictions = []
        for i in range(len(X)):
            px_y_spam = []
            px_y_non_spam = []
            for j in range(len(X[0])):
                val = X[i][j]
                px_y_non_spam.append(count_table_X0[j][int(val - 1)] / gtm_non_spam_len)
                px_y_spam.append(count_table_X1[j][int(val - 1)] / gtm_spam_len)
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
        print(accuracy)
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
