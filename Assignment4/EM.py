import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal


class EM:
    data = []
    M = 2
    mean = []
    sigma = []
    pi = []
    Z = []


    def read(self):
        self.data = pd.read_csv('data4-1.data')
        for i in range(self.M):
            self.mean.append(np.random.random_sample((2,)))
            self.sigma.append(np.identity(2))
            self.pi.append(len(self.data)/2)

    def train(self):
        X = self.data
        self.Z = np.zeros((len(self.data), 2))
        for e in range(20):
            #E step
            px = []
            for m in range(self.M):
                px.append(multivariate_normal.pdf(self.data, mean=self.mean[m], cov=self.sigma[m]))
            for i in range(len(self.data)):
                total = 0
                for m in range(self.M):
                    total = total + px[m][i]*self.pi[m]
                for m in range(self.M):
                    self.Z[i][m] = (px[m][i]*self.pi[m])/total
            #M step
            for m in range(self.M):
                sigma_sum = 0
                den = 0
                mean_sum = 0
                X = self.data.values
                for i in range(len(self.data)):
                    a = self.Z[i][m]
                    X[i] = X[i].reshape((1, 2))
                    _mean = self.mean[m]
                    _mean = _mean.reshape((1, 2))
                    # b = np.mat(X[i] - self.mean[m])
                    b = X[i] - _mean
                    sigma_sum = sigma_sum + a*np.dot(b.T,b)
                    den = den + self.Z[i][m]
                    mean_sum = mean_sum + self.Z[i][m]*X[i]
                self.sigma[m] = sigma_sum/den
                self.mean[m] = mean_sum/den
                self.pi[m] = den/len(self.data)

    def test(self):
        print(self.mean[0])
        print(self.mean[1])
        print(self.sigma[0])
        print(self.sigma[1])

        n1 = 0
        n2 = 0
        for row in self.Z:
            if row[0] > row[1]:
                n1 = n1+1
            else:
                n2 = n2+1
        print(n1)
        print(n2)


def main():
    em = EM()
    em.read()
    em.train()
    em.test()

if __name__ == '__main__':
    main()

