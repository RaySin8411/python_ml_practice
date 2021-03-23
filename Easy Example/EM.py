import math
import copy
import numpy as np
import matplotlib.pyplot as plt

isdebug = False


class EMAlgorithm(object):

    def __init__(self, Sigma=6, Mu1=40, Mu2=20, k=2, N=1000, iter_num=1000, Epsilon=0.1):
        self.Sigma = Sigma
        self.Mu1 = Mu1
        self.Mu2 = Mu2
        self.k = k
        self.N = N
        self.iter_num = iter_num
        self.Epsilon = Epsilon

    def ini_data(self, Sigma, Mu1, Mu2, k, N):
        global X
        global Mu
        global Expectations
        X = np.zeros((1, N))
        Mu = np.random.random(2)
        Expectations = np.zeros((N, k))
        for i in range(0, N):
            if np.random.random(1) > 0.5:
                X[0, i] = np.random.normal() * Sigma + Mu1
            else:
                X[0, i] = np.random.normal() * Sigma + Mu2
        if isdebug:
            print("**************")
            print("初始觀測數據X:")
            print(X)

    # EM算法 : 步驟1, 計算E[Zij]
    def e_step(self, Sigma, k, N):
        global Expectations
        global Mu
        global X

        for i in range(0, N):
            Denom = 0
            for j in range(0, k):
                Denom += math.exp((-1 / 2 * (float(Sigma ** 2)))) * (float(X[0, i] - Mu[j]))
            for j in range(0, k):
                Numer = math.exp((-1 / (2 * (float(Sigma ** 2)))) * (float(X[0, i] - Mu[j])) ** 2)
                Expectations[i, j] = Numer / Denom
            if isdebug:
                print("**************")
                print("隱藏變量E(Z):")
                print(Expectations)

    # EM算法 : 步驟2, 求最大化E[Zij]的參數Mu
    def m_step(self, k, N):
        global Expectations
        global X
        for j in range(0, k):
            Numer = 0
            Denom = 0
            for i in range(0, N):
                Numer += Expectations[i, j] * X[0, i]
                Denom += Expectations[i, j]
            Mu[j] = Numer / Denom

    # 算法迭代iter_num次, 或達到精度Epsilon停止迭代
    def run(self, Sigma, Mu1, Mu2, k, N, iter_num, Epsilon):
        self.ini_data(Sigma, Mu1, Mu2, k, N)
        print("初始<u1,u2>", Mu)
        for i in range(iter_num):
            Old_Mu = copy.deepcopy(Mu)
            self.e_step(Sigma, k, N)
            self.m_step(k, N)
            print(i, Mu)
            if sum(abs(Mu - Old_Mu)) < Epsilon:
                break


class Em:
    def gauss_distribution(self, datain, miu, data):
        return (np.exp(-(datain - miu) ** 2 / 2 / data)) / np.sqrt(2 * np.pi * data)

    def Rjk(self, alpha, gauss):
        rjk = (alpha * gauss).T / np.sum(alpha * gauss, axis=1)
        return rjk

    def main(self, datain, k, epoch):
        N = np.shape(datain)[0]
        alpha = np.zeros(k) + [0.8, 0.2]
        miu = np.zeros(k) + [50, 20]
        deta = np.ones(k) + [5, 15]
        datain1 = np.ones((N, k))
        for i in range(k):
            datain1[:, i] = datain
        for j in range(epoch):
            gauss = self.gauss_distribution(datain1, miu, deta)
            rij = self.Rjk(alpha, gauss)
            miu = np.sum(rij * datain, axis=1) / np.sum(rij, axis=1)
            deta = np.sum(rij * (datain1 - miu).T ** 2, axis=1) / np.sum(rij, axis=1)
            alpha = np.sum(rij, axis=1) / N
            print('the %d step:alpha=%s miu=%s deta=%s' % (j, miu, deta, alpha))
        return (alpha, miu, deta)


def main():
    x = EMAlgorithm()
    x.run(6, 40, 20, 2, 1000, 1000, 0.01)
    plt.hist(X[0, :], 50)
    plt.show()

    datain = [-64, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]
    x = Em()
    alpha, miu, deta = x.main(datain, 2, 60)

    plt.figure(figsize=(8, 5), dpi=80)
    plt.subplot(111)

    plt.hist(datain)
    ax = plt.gca()

    fangda = 230
    x = range(-80, 80, 1)
    y0 = np.exp(-(x - miu[0]) ** 2 / 2 / deta[0]) / np.sqrt(2 * np.pi * deta[0])
    y1 = np.exp(-(x - miu[1]) ** 2 / 2 / deta[1]) / np.sqrt(2 * np.pi * deta[1])
    line1 = ax.plot(x, y0 * alpha[0] * fangda, c='y', linewidth=2.5)
    line2 = ax.plot(x, y1 * alpha[1] * fangda, c='b', linewidth=2.5)
    line3 = ax.plot(x, (y0 * alpha[0] + y1 * alpha[1]) * fangda, c='r', linewidth=2.5)
    plt.show()


if __name__ == "__main__":
    main()
