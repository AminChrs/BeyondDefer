# testing H(Y|X) and H(Y|X1, X2) for X1 = XOR(Y, n_1) and X2 = XOR(Y, n2)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import random
import time
import scipy.stats as stats
import scipy.special as special
import scipy.optimize as optimize

def H(Y, X, num_symbol_Y, num_symbol_X):
    # H(Y|X) = H(Y,X) - H(X)
    return H_Y_X(Y, X, num_symbol_Y, num_symbol_X) - H_X(X, num_symbol_X)

def H_Y_X(Y, X, num_symbol_Y, num_symbol_X):
    # find joint distribution of Y and X
    p_Y_X = joint_distribution(Y, X, num_symbol_Y, num_symbol_X)
    # find entropy of joint distribution
    return entropy(p_Y_X)

def H_X(X, num_symbol_X):
    # find distribution of X
    p_X = distribution(X, num_symbol_X)
    # find entropy of distribution
    return entropy(p_X)

def joint_distribution(Y, X, num_symbol_Y, num_symbol_X):
    # find joint distribution of Y and X
    p_Y_X = np.zeros((num_symbol_Y, num_symbol_X))
    for i in range(len(Y)):
        p_Y_X[Y[i], X[i]] += 1
    p_Y_X = p_Y_X / len(Y)
    return p_Y_X

def distribution(X, num_symbol_X):
    # find distribution of X
    p_X = np.zeros(num_symbol_X)
    for i in range(len(X)):
        p_X[X[i]] += 1
    p_X = p_X / len(X)
    return p_X

def entropy(p):
    # find entropy of distribution
    # if p is 0, then log2(p) is -inf, so we need to replace -inf with 0
    logs = np.log2(p)
    isinf = np.isinf(logs)
    # print(logs)
    logs[isinf] = 0.0
    # oneminuslogs = np.log2(1-p)
    # isinf = np.isinf(oneminuslogs)
    # oneminuslogs[isinf] = 0.0
    return -np.sum(p * logs)


# maximum likelihood estimation based on the joint distribution
def MLE(X, p_Y_X):

    # find the marginal distribution of X based on joint distribution p_Y_X
    p_X = np.sum(p_Y_X, axis=0)

    # find conditional distribution of Y given X
    p_Y_X_cond = p_Y_X / p_X

    Y = np.zeros(len(X))

    # print("Shape of p_Y_X_cond = ", p_Y_X_cond.shape)

    for i in range(len(X)):
        # find the most likely Y given X
        # print("X[i] = ", X[i])
        # print("P(Y|X) = ", p_Y_X_cond[:, X[i]])
        Y[i] = np.argmax(p_Y_X_cond[:, X[i]])

    return Y

# accuracy of the estimation

def accuracy(Y, Y_est):
    
        # find the number of correct estimation
        correct = np.sum(Y == Y_est)
    
        # find the accuracy
        return correct / len(Y)


def accs(p, q, num_symbol_Y):

    # number of samples
    n = 100000

    # number of symbols
    # num_symbol_Y = 3

    # generate random bits for Y with probability q to be 1
    Y = np.random.choice(np.arange(0, num_symbol_Y), size=n, p=q)
    # Y = np.random.randint(2, size=n)

    # p = 0.3
    # n1 and n2 are random bits with probability 0.1 to be 1
    n_1 = np.random.choice(np.arange(0, num_symbol_Y), size=n, p=p)
    n_2 = np.random.choice(np.arange(0, num_symbol_Y), size=n, p=p)

    # generate X1 and X2 by adding Y and ni mode num_symbol_Y
    X_1 = (Y + n_1) % num_symbol_Y
    X_2 = (Y + n_2) % num_symbol_Y

    # entangle X1 and Y as a tuple
    X_1_Y = np.zeros((n, 2))
    X_1_Y[:, 0] = X_1
    X_1_Y[:, 1] = Y
    # make sure everything is printed
    np.set_printoptions(threshold=sys.maxsize)
    # print("X_1_Y = ", X_1_Y)
    # turn X1 and X2 to a single array of numbers between 0 to 3
    X = np.zeros(n)
    for i in range(n):
        X[i] = X_1[i] + num_symbol_Y * X_2[i]
    # convert X to int
    X = X.astype(int)
    # find H(Y|X)
    ent_Y_X = H(Y, X, num_symbol_Y, num_symbol_Y**2)
    # print("H(Y|X) = ", ent_Y_X)

    # find H(Y|X1)
    ent_Y_X1 = H(Y, X_1.astype(int), num_symbol_Y, num_symbol_Y**2)
    # print("H(Y|X1) = ", ent_Y_X1)

    # find the distribution
    p_Y_X = joint_distribution(Y, X, num_symbol_Y, num_symbol_Y**2)

    # print("Joint distribution of Y and X = ", p_Y_X)

    # find MLE of Y given X
    Y_est = MLE(X, p_Y_X)

    Y_diff = np.logical_xor(Y, Y_est)
    # find mutual information of Y_diff and Y
    ent_Y_diff_Y = H(Y_diff.astype(int), X.astype(int), num_symbol_Y, num_symbol_Y**2)
    ent_Y = H_X(Y_diff.astype(int), num_symbol_Y)
    print("H(Y_diff|Y) = ", ent_Y_diff_Y)
    print("H(Y) = ", ent_Y)
    MI_Y_Y_diff_joint = ent_Y - ent_Y_diff_Y
    # find the accuracy
    acc_joint = accuracy(Y, Y_est)
    # print("Accuracy of Y|X1, X2 = ", acc_joint)
    ent_err_joint = entropy([1-acc_joint, acc_joint])

    # find MLE of Y given X1

    p_Y_X1 = joint_distribution(Y, X_1.astype(int), num_symbol_Y, num_symbol_Y)
    # print("Joint distribution of Y and X1 = ", p_Y_X1)
    Y_est = MLE(X_1.astype(int), p_Y_X1)

    # find the accuracy
    acc_X1 = accuracy(Y, Y_est)
    # print("Accuracy of Y|X1 = ", acc_X1)

    # find entropy of error of Y_est

    ent_err_X1 = entropy([1-acc_X1, acc_X1])

    # find mutual information of Y\neq Y_est and Y
    # find Y\neq Y_est
    Y_diff = np.logical_xor(Y, Y_est)
    # find mutual information of Y_diff and Y
    ent_Y_diff_X = H(Y_diff.astype(int), X_1.astype(int), num_symbol_Y, num_symbol_Y)
    ent_Y = H_X(Y_diff.astype(int), num_symbol_Y)
    MI_X_1_Y_diff_X1 = ent_Y - ent_Y_diff_X

    # 

    class return_class:
        def __init__(self):
            self.acc_joint = []
            self.acc_X1 = []
            self.ent_Y_X = []
            self.ent_Y_X1 = []
            self.ent_err_joint = []
            self.ent_err_X1 = []
            self.MI_X_1_Y_diff_X1 = []
            self.MI_Y_Y_diff_joint = []
    ret = return_class()
    ret.acc_joint = acc_joint
    ret.acc_X1 = acc_X1
    ret.ent_Y_X = ent_Y_X
    ret.ent_Y_X1 = ent_Y_X1
    ret.ent_err_joint = ent_err_joint
    ret.ent_err_X1 = ent_err_X1
    ret.MI_X_1_Y_diff_X1 = MI_X_1_Y_diff_X1
    ret.MI_Y_Y_diff_joint = MI_Y_Y_diff_joint
    return ret



def main():
    p = 0.4
    q_list = np.arange(0.03, 1.0, 0.03)

    acc_joint_list = np.zeros(len(q_list))
    acc_X1_list = np.zeros(len(q_list))
    ent_Y_X_list = np.zeros(len(q_list))
    ent_Y_X1_list = np.zeros(len(q_list))
    ent_err_joint_list = np.zeros(len(q_list))
    ent_err_X1_list = np.zeros(len(q_list))
    MI_Y_Y_diff_X1_list = np.zeros(len(q_list))
    MI_Y_Y_diff_joint_list = np.zeros(len(q_list))

    for i in range(len(q_list)):
        num_symbol_Y = 3
        # generate num_symbol_Y-dimensional array p out of p by attaching (1-p)/2 for num_symbol_Y-1 times to the left and right
        p_n = np.hstack((np.ones(num_symbol_Y-1)*(p)/(num_symbol_Y-1), 1-p))
        # generate array q out of q
        p_Y = np.hstack((np.ones(num_symbol_Y-1)*(q_list[i])/(num_symbol_Y-1), 1-q_list[i]))
        acc = accs(p_n, p_Y, num_symbol_Y)
        acc_joint_list[i] = acc.acc_joint
        acc_X1_list[i] = acc.acc_X1
        ent_Y_X_list[i] = acc.ent_Y_X
        ent_Y_X1_list[i] = acc.ent_Y_X1
        ent_err_joint_list[i] = acc.ent_err_joint
        ent_err_X1_list[i] = acc.ent_err_X1
        MI_Y_Y_diff_X1_list[i] = acc.MI_X_1_Y_diff_X1
        MI_Y_Y_diff_joint_list[i] = acc.MI_Y_Y_diff_joint

    plt.plot(q_list, acc_joint_list, label="Y|X1, X2")
    plt.plot(q_list, acc_X1_list, label="Y|X1")
    plt.xlabel("q")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.figure()
    plt.plot(q_list, ent_Y_X_list, label="H(Y|X1, X2)")
    plt.plot(q_list, ent_Y_X1_list, label="H(Y|X1)")
    plt.plot(q_list, ent_err_joint_list, label="H(Error joint)")
    plt.plot(q_list, ent_err_X1_list, label="H(Error X1)")
    # plt.plot(q_list, -MI_Y_Y_diff_X1_list+ent_err_X1_list, label="Est H(Y|X1)")
    # plt.plot(q_list, -MI_Y_Y_diff_joint_list+ent_err_joint_list, label="Est H(Y|X1, X2)")

    plt.xlabel("q")
    plt.ylabel("Entropy")
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    main()