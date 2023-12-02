# Write a code to generate a noisy version of a binary random variable by XORing it with a random bit.

import random
from matplotlib import pyplot as plt
import numpy as np

def noisy_binary_random_variable(y, p):
    # the random bit is Bernoulli distributed with p
    if random.random() < p:
        return y ^ 1
    else:
        return y
# a loss function that if y = 0 and f = 1 it has cost of 1-c while if y = 1 and f = 0 it has cost of c
def loss_function(y, f, c):
    if y == 0 and f == 1:
        return 1 - c
    elif y == 1 and f == 0:
        return c
    else:
        return 0

# expected loss function
def expected_loss(f, y, c):
    return sum([loss_function(y[i], f[i], c) for i in range(len(y))]) / len(y)
# define a function that returns binary representation of a number
def binary_representation(x):
    # make sure that the list is of length 4
    return [int(i) for i in list('{0:04b}'.format(x))]

# a function that converts two binaries to a decimal number j and  takes i as input and find jth element of the list that is the output of the binary representation of i
def bin_fun(x1, x2, j):
    i = 2 *x2 + x1
    return binary_representation(j)[i]

# makes the above function to work for a list
def bin_fun_list(x1, x2, j):
    return [bin_fun(x1[i], x2[i], j) for i in range(len(x1))]

def test_def_and_min(N, N_test, p1, p2, q, c):
    # generate N number of y that takes 1 with probability q
    y = [1 if random.random() < q else 0 for i in range(N)]
    h = [noisy_binary_random_variable(y[i], p1) for i in range(N)]
    M = [noisy_binary_random_variable(y[i], p2) for i in range(N)]
    # print binary representation of the 4
    # print('Binary representation of 4 is {}'.format(binary_representation(4)))
    loss_f = []
    for i in range(16):
        f = bin_fun_list(h, M, i)
        loss_f.append(expected_loss(f, y, c))
        # print('Expected loss for f{} is {}'.format(i, expected_loss(f, y, c)))
    loss_h = expected_loss(h, y, c)
    loss_M = expected_loss(M, y, c)
    # find the argmin of loss
    min_loss = loss_f.index(min(loss_f))
    # deferral is 0 if loss of M is more than loss of h and 1 otherwise
    deff = 1 if loss_M < loss_h else 0
    # if deff == 1:
    #     # print deferral loss of M
    #     # print('Deferral loss of M is {}'.format(loss_M))
    # else:
        # print deferral loss of h
        # print('Deferral loss of h is {}'.format(loss_h))

    # test with new y, h, and M
    # print("N_test is {}".format(N_test))
    y_test = [1 if random.random() < q else 0 for i in range(N_test)]
    h_test = [noisy_binary_random_variable(y_test[i], p1) for i in range(N_test)]
    M_test = [noisy_binary_random_variable(y_test[i], p2) for i in range(N_test)]
    # find loss of h and M
    loss_h_test = expected_loss(h_test, y_test, c)
    loss_M_test = expected_loss(M_test, y_test, c)
    # find the deferral loss
    if deff == 1:
        def_loss = loss_M_test
    else:
        def_loss = loss_h_test

    # find the loss of f with the argmin
    for i in range(16):
        f = bin_fun_list(h_test, M_test, i)
        # print('Expected loss for f{} is {}'.format(i, expected_loss(f, y_test, c)))
    f_test = bin_fun_list(h_test, M_test, min_loss)
    loss_f_test = expected_loss(f_test, y_test, c)
    # print f_test and loss_f_test
    # print('Loss min is {}'.format(loss_f_test))
    # print('Loss of deferral is {}'.format(def_loss))
    return loss_f_test, def_loss
# main function
def main():
    N_iter = 100
    # generate N_iter number of q and p1 and p2 randomly between 0 and 1/2
    q = [random.random() / 2 for i in range(N_iter)]
    p1 = [random.random() / 2 for i in range(N_iter)]
    p2 = [random.random() / 2 for i in range(N_iter)]
    # generate c randomly between 0 and 1
    c = [random.random() for i in range(N_iter)]
    # generate N between 10 and 1000 with step 10
    N = np.arange(1, 200, 10)
    N_test = 1000
    # find the average and the standard deviation of the result of test_def_and_min for all values of N
    avg_loss_f_test = []
    std_loss_f_test = []
    avg_def_loss = []
    std_def_loss = []
    for j in range(len(N)):
        # show a progress bar
        print('Progress: {}%'.format(j / len(N) * 100))
        loss_f_test = []
        def_loss = []
        for i in range(N_iter):
            l_f, l_def = test_def_and_min(N[j], N_test, p1[i], p2[i], q[i], c[i])
            loss_f_test.append(l_f)
            def_loss.append(l_def)
        avg_loss_f_test.append(sum(loss_f_test) / len(loss_f_test))
        std_loss_f_test.append(np.std(loss_f_test))
        avg_def_loss.append(sum(def_loss) / len(def_loss))
        std_def_loss.append(np.std(def_loss))
    # plot the results
    # make the lines thicker
    plt.figure()
    # make error bars having a hat shape
    plt.rcParams['errorbar.capsize'] = 10
    plt.errorbar(N, avg_loss_f_test, yerr=std_loss_f_test, label = 'Optimal Trained Operation Loss', linewidth=4.0, elinewidth=2.0)
    # plt.errorbar(N, avg_loss_f_test, yerr=std_loss_f_test, label = 'Optimal Trained Operation Loss', linewidth=4.0)
    # plt.xlabel('N')
    plt.errorbar(N, avg_def_loss, yerr=std_def_loss, label = 'Deferral Loss', linewidth=4.0, elinewidth=2.0)
    plt.xlabel('Training Set Size')
    plt.ylabel('Expected Loss')
    # plt.title('Average deferral loss vs N')
    plt.legend()
    # plt.show()
    # save the figure
    plt.savefig('deferral_loss_vs_N.pdf')
    plt.show()
if __name__ == '__main__':
    main()