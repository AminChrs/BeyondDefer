# read json file 'results_is_defer='+str(p)+'_k='+str(k)+'_c='+str(c)+'.json' where p is a boolean and k and c are elements of a list k_list and c_list

import json
import numpy as np
a_list = [0.0, 0.5, 1.0]
k_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
c_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
p = False
test_size = 5000
#make an array acc_hybrid with size of len(c_list)
acc_hybrid = np.zeros([len(c_list), len(k_list)])
cov_hybrid = np.zeros([len(c_list), len(k_list)])
cost_hybrid = np.zeros([len(c_list), len(k_list)])
mean_acc_hybrid = np.zeros([len(c_list),])
mean_cov_hybrid = np.zeros([len(c_list),])
mean_cost_hybrid = np.zeros([len(c_list),])
std_acc_hybrid = np.zeros([len(c_list),])
std_cov_hybrid = np.zeros([len(c_list),])
std_cost_hybrid = np.zeros([len(c_list),])
for c in c_list:
    # find the index of c in c_list
    index = c_list.index(c)
    for k in k_list:
        index_k = k_list.index(k)
        with open('files/results_is_defer='+str(p)+'_k='+str(k)+'_c='+str(c)+'.json') as f:
            data = json.load(f)
        acc_hybrid[index, index_k] += data['hybrid']['accuracy']
        cov_hybrid[index, index_k] += data['hybrid']['coverage']
        cost_hybrid[index, index_k] += (test_size - data['hybrid']['coverage'])/test_size*c+(1-data['hybrid']['accuracy']/100)
    mean_acc_hybrid[index] = np.mean(acc_hybrid[index, :])
    mean_cov_hybrid[index] = np.mean(cov_hybrid[index, :])
    mean_cost_hybrid[index] = np.mean(cost_hybrid[index, :])
    std_acc_hybrid[index] = np.std(acc_hybrid[index, :])
    std_cov_hybrid[index] = np.std(cov_hybrid[index, :])
    std_cost_hybrid[index] = np.std(cost_hybrid[index, :])

p=True

acc_defer = np.zeros([len(a_list), len(c_list), len(k_list)])
cov_defer = np.zeros([len(a_list), len(c_list), len(k_list)])
cost_defer = np.zeros([len(a_list), len(c_list), len(k_list)])
mean_acc_defer = np.zeros([len(a_list), len(c_list),])
mean_cov_defer = np.zeros([len(a_list), len(c_list),])
mean_cost_defer = np.zeros([len(a_list), len(c_list),])
std_acc_defer = np.zeros([len(a_list), len(c_list),])
std_cov_defer = np.zeros([len(a_list), len(c_list),])
std_cost_defer = np.zeros([len(a_list), len(c_list),])
for a in a_list:
    index_a = a_list.index(a)
    for c in c_list:
        index_c = c_list.index(c)
        for k in k_list:
            index_k = k_list.index(k)
            with open('files/results_is_defer='+str(p)+'_k='+str(k)+'_a='+str(a)+'_c='+str(c)+'.json') as f:
                data = json.load(f)
            acc_defer[index_a, index_c, index_k] += data['deferral']['metrics']['system accuracy']
            # find the first word in the string data['deferral']['metrics']['coverage'] and convert to int
            cov = int(data['deferral']['metrics']['coverage'].split()[0])
            cov_defer[index_a, index_c, index_k] += cov
            cost_defer[index_a, index_c, index_k] += (test_size - cov)/test_size*c+(1-data['deferral']['metrics']['system accuracy']/100)
        mean_acc_defer[index_a, index_c] = np.mean(acc_defer[index_a, index_c, :])
        mean_cov_defer[index_a, index_c] = np.mean(cov_defer[index_a, index_c, :])
        mean_cost_defer[index_a, index_c] = np.mean(cost_defer[index_a, index_c, :])
        std_acc_defer[index_a, index_c] = np.std(acc_defer[index_a, index_c, :])
        std_cov_defer[index_a, index_c] = np.std(cov_defer[index_a, index_c, :])
        std_cost_defer[index_a, index_c] = np.std(cost_defer[index_a, index_c, :])

# plot the mean and std of accuracy, coverage and cost for defer and hybrid

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
for i in range(len(a_list)):
    plt.errorbar(c_list, mean_acc_defer[i, :], yerr=std_acc_defer[i, :], label=r'Deferral for $\alpha$='+str(a_list[i]))
plt.errorbar(c_list, mean_acc_hybrid, yerr=std_acc_hybrid, label='Hybrid')
plt.xlabel('Cost of deferral')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(3, 1, 2)
for i in range(len(a_list)):
    plt.errorbar(c_list, mean_cov_defer[i, :]/5000, yerr=std_cov_defer[i, :]/5000, label=r'Deferral for $\alpha$='+str(a_list[i]))

plt.errorbar(c_list, mean_cov_hybrid/5000, yerr=std_cov_hybrid/5000, label='Hybrid')
plt.xlabel('Cost of deferral')
plt.ylabel('Coverage')
plt.legend()
plt.subplot(3, 1, 3)

for i in range(len(a_list)):
    plt.errorbar(c_list, mean_cost_defer[i, :], yerr=std_cost_defer[i, :], label=r'Deferral for $\alpha$='+str(a_list[i]))
plt.errorbar(c_list, mean_cost_hybrid, yerr=std_cost_hybrid, label='Hybrid')
plt.xlabel('Cost of deferral')
plt.ylabel('Overal loss')
plt.legend()
plt.show()

# save the figure
plt.savefig('results.png')
