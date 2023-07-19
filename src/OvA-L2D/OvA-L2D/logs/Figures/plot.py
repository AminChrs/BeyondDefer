from __future__ import division
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy import stats
from newCalibration import Tensorize
from newCalibration import TempScaling
from matplotlib import rc
import seaborn as sns
plt.rcParams['xtick.labelsize'] = 42
plt.rcParams['ytick.labelsize'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
rc('font', family='serif')
rc('text', usetex=True)

# === Matplotlib Options === #
plot_args = {"linestyle": "-",
                "marker": "o",
                "markeredgecolor": "k",
                "markersize": 10,
                "linewidth": 4
                }

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 30})



path_defer = './../../Sontag_Results_Ks_Experiment/expert_predict_'
path_ova = './../../OVA_Results_Ks_Experiment/expert_predict_'

path_defer = './../Data/Sontag_Results_Ks_Experiment/expert_predict_'
path_ova = './../Data/OVA_Results_Ks_Experiment/expert_predict_'





system_accuracy_defer = []
coverage_defer = []
accuracy_clf_defer = []
for seed in [948,  625,  436,  791, 1750]:
    system_accuracy_defer_ = []
    coverage_defer_ = []
    accuracy_clf_defer_ = []

    with open(path_defer + 'validation_alpha_1.0_seed_' + str(seed) + '.txt', 'r') as f:
        ce = json.load(f)

    for k, v in ce.items():
        system_accuracy_defer_.append(v['cifar']['system_accuracy'])
        cov = v['cifar']['coverage'].split(' out of')
        a, b = cov[0], cov[1]
        coverage_defer_.append(int(a)/int(b))
        accuracy_clf_defer_.append(float(v['cifar']['classifier_accuracy']))
    system_accuracy_defer.append(system_accuracy_defer_)
    coverage_defer.append(coverage_defer_)
    accuracy_clf_defer.append(accuracy_clf_defer_)

def read(seed, path = path_ova):
    system_accuracy_ = []
    coverage = []
    classifier_acc_ = []

    severity_levels = ['cifar']

    sigmoid = nn.Sigmoid()

    for k in [2,4,6,8]:
        with open(path+'logits_alpha_seed_' + str(seed) + '.txt', 'r') as f:
            data = json.loads(json.load(f))[str(k)]
        
        with open(path + 'validation_seed_' + str(seed) + '.txt', 'r') as f:
            val = json.load(f)[str(k)]

        with open(path + 'true_label_seed_' + str(seed) + '.txt', 'r') as f:
            true = json.loads(json.load(f))[str(k)]
        
        with open(path + 'expert_predictions_seed_' + str(seed) + '.txt', 'r') as f:
            expert = json.loads(json.load(f))[str(k)]

        cov_ = {}
        clf_acc_ = {}
        system_acc_ = {}
        for severity in severity_levels:
            logits = Tensorize(data[severity])
            true_labels = Tensorize(true[severity])
            expert_labels = Tensorize(expert[severity])
            probs = sigmoid(logits)
            pred_labels = torch.argmax(probs, dim=1)

            mask = torch.gt(logits, torch.tensor([0])).float()
            temp = torch.sum(mask, dim=1, keepdim=True)
            none = torch.where(temp == 0.0)[0]

            ones = torch.where(temp == 1.0)
            map_ = {}
            for i,v in enumerate(ones[0]):
                map_[i] = v.item()
            logits_ones = logits[ones[0]]
            _, ids = torch.max(logits_ones, dim=1)
            rej_ones = torch.where(ids == 10)[0]
            clf_ones = torch.where(ids != 10)[0]

            rej_ids = []
            clf_ids = []
            for i in rej_ones:
                rej_ids.append(map_[i.item()])
            
            for i in clf_ones:
                clf_ids.append(map_[i.item()])

            mask2 = torch.gt(logits[:,:10], torch.tensor([0])).float()
            temp2 = torch.sum(mask2, dim=1, keepdim=True)
            ambiguous_2 = torch.where(temp2 == 2.0)[0]

            ambiguous = set(ambiguous_2.tolist())

            all = set(none.tolist()).union(set(ones[0].tolist())).union(ambiguous)
            remaining = set(range(logits.shape[0])) - all

            map2 = {}
            for i, j in enumerate(list(remaining)):
                map2[i] = j
            rem = logits[list(remaining)]
            assert torch.sum(torch.gt(rem, torch.tensor([0]).float())).tolist() == 2.0*len(remaining)

            _, p = torch.max(rem, dim=1)
            rej_ = torch.where(p == 10)[0]
            clf_ = torch.where(p != 10)[0]

            r = []
            for i in rej_:
                r.append(map2[i.item()])
            
            c = []
            for i in clf_:
                c.append(map2[i.item()])

            #indices where the expert is predicting
            rej = set(rej_ids).union(set(r))
            #indices where the classifier is predicting
            clf = set(clf_ids).union(set(c))

            reject_idx = list(rej)
            clf_idx = list(clf)

            clf_pred = pred_labels[clf_idx]
            clf_accuracy = np.mean(np.equal(clf_pred.numpy(), true_labels[clf_idx].numpy()).astype(int))

            expert_pred = expert_labels[reject_idx]
            expert_accuracy = np.mean(np.equal(expert_pred.numpy(), true_labels[reject_idx].numpy()).astype(int))

            system_accuracy = np.equal(clf_pred.numpy(), true_labels[clf_idx].numpy()).astype(int).tolist()
            system_accuracy.extend(np.equal(expert_pred.numpy(), true_labels[reject_idx].numpy()).astype(int).tolist())

            sys_acc = np.average(system_accuracy)
            


            cov_[severity] =  len(clf)/logits.shape[0]
            clf_acc_[severity] = clf_accuracy
            system_acc_[severity] = sys_acc
        
        columns1 = [clf_acc_[severity] for severity in severity_levels]
        classifier_acc_.append(columns1)
        columns2 = [cov_[severity] for severity in severity_levels]
        coverage.append(columns2)
        columns3 = [system_acc_[severity] for severity in severity_levels]
        system_accuracy_.append(columns3)

    return classifier_acc_, coverage, system_accuracy_



def read_calibrated(seed, path = path_ova):
    system_accuracy_ = []
    coverage = []
    classifier_acc_ = []

    severity_levels = ['cifar']

    sigmoid = nn.Sigmoid()

    for k in [2,4,6,8]:
        with open(path+'logits_alpha_seed_' + str(seed) + '.txt', 'r') as f:
            data = json.loads(json.load(f))[str(k)]
        
        with open(path + 'validation_seed_' + str(seed) + '.txt', 'r') as f:
            val = json.load(f)[str(k)]

        with open(path + 'true_label_seed_' + str(seed) + '.txt', 'r') as f:
            true = json.loads(json.load(f))[str(k)]
        
        with open(path + 'expert_predictions_seed_' + str(seed) + '.txt', 'r') as f:
            expert = json.loads(json.load(f))[str(k)]

        cov_ = {}
        clf_acc_ = {}
        system_acc_ = {}
        for severity in severity_levels:
            print("-------")
            calibrator = TempScaling()
            calibrator.load_state_dict(torch.load('./../Calibrator_New/Temp_Scaling_K_' + str(k) + '_seed_' + str(seed) + '.pt'))
            logits = Tensorize(data[severity])
            true_labels = Tensorize(true[severity])
            expert_labels = Tensorize(expert[severity])
            cp_ = calibrator(logits[:,10])
            logits[:,10] = cp_
            probs = sigmoid(logits)
            pred_labels = torch.argmax(probs, dim=1)

            mask = torch.gt(logits, torch.tensor([0])).float()
            temp = torch.sum(mask, dim=1, keepdim=True)
            none = torch.where(temp == 0.0)[0]


            ones = torch.where(temp == 1.0)
            map_ = {}
            for i,v in enumerate(ones[0]):
                map_[i] = v.item()
            logits_ones = logits[ones[0]]
            _, ids = torch.max(logits_ones, dim=1)
            rej_ones = torch.where(ids == 10)[0]
            clf_ones = torch.where(ids != 10)[0]

            rej_ids = []
            clf_ids = []
            for i in rej_ones:
                rej_ids.append(map_[i.item()])
            
            for i in clf_ones:
                clf_ids.append(map_[i.item()])

            mask2 = torch.gt(logits[:,:10], torch.tensor([0])).float()
            temp2 = torch.sum(mask2, dim=1, keepdim=True)
            ambiguous_2 = torch.where(temp2 == 2.0)[0]

            ambiguous = set(ambiguous_2.tolist()) 

            all = set(none.tolist()).union(set(ones[0].tolist())).union(ambiguous)
            remaining = set(range(logits.shape[0])) - all

            map2 = {}
            for i, j in enumerate(list(remaining)):
                map2[i] = j
            rem = logits[list(remaining)]
            assert torch.sum(torch.gt(rem, torch.tensor([0]).float())).tolist() == 2.0*len(remaining)

            _, p = torch.max(rem, dim=1)
            rej_ = torch.where(p == 10)[0]
            clf_ = torch.where(p != 10)[0]

            r = []
            for i in rej_:
                r.append(map2[i.item()])
            
            c = []
            for i in clf_:
                c.append(map2[i.item()])

            #indices where the expert is predicting
            rej = set(rej_ids).union(set(r))
            #indices where the classifier is predicting
            clf = set(clf_ids).union(set(c))

            reject_idx = list(rej)
            clf_idx = list(clf)

            clf_pred = pred_labels[clf_idx]
            clf_accuracy = np.mean(np.equal(clf_pred.numpy(), true_labels[clf_idx].numpy()).astype(int))

            expert_pred = expert_labels[reject_idx]
            expert_accuracy = np.mean(np.equal(expert_pred.numpy(), true_labels[reject_idx].numpy()).astype(int))

            system_accuracy = np.equal(clf_pred.numpy(), true_labels[clf_idx].numpy()).astype(int).tolist()
            system_accuracy.extend(np.equal(expert_pred.numpy(), true_labels[reject_idx].numpy()).astype(int).tolist())

            sys_acc = np.average(system_accuracy)
            


            cov_[severity] =  len(clf)/logits.shape[0]  
            clf_acc_[severity] = clf_accuracy
            system_acc_[severity] = sys_acc
        
        columns1 = [clf_acc_[severity] for severity in severity_levels]
        classifier_acc_.append(columns1)
        columns2 = [cov_[severity] for severity in severity_levels]
        coverage.append(columns2)
        columns3 = [system_acc_[severity] for severity in severity_levels]
        system_accuracy_.append(columns3)

    return classifier_acc_, coverage, system_accuracy_

def flat(l, scale=1):
    return [i*scale for sublist in  l for i in sublist]

accuracy_clf_ova = []
coverage_ova = []
system_accuracy_ova = []
for seed in [948,  625,  436,  791, 1750]:
    accuracy_clf_ova_, coverage_ova_, system_accuracy_ova_ = read(seed, path=path_ova)
    accuracy_clf_ova.append(flat(accuracy_clf_ova_, scale=100))
    coverage_ova.append(flat(coverage_ova_))
    system_accuracy_ova.append(flat(system_accuracy_ova_, scale=100))


accuracy_clf_ova_c = []
coverage_ova_c = []
system_accuracy_ova_c = []
for seed in [948,  625,  436,  791, 1750]:
    accuracy_clf_ova_, coverage_ova_, system_accuracy_ova_ = read_calibrated(seed, path=path_ova)
    accuracy_clf_ova_c.append(flat(accuracy_clf_ova_, scale=100))
    coverage_ova_c.append(flat(coverage_ova_))
    system_accuracy_ova_c.append(flat(system_accuracy_ova_, scale=100))



accuracy_clf_defer = np.array(accuracy_clf_defer)
coverage_defer = np.array(coverage_defer)
system_accuracy_defer = np.array(system_accuracy_defer)

accuracy_clf_ova = np.array(accuracy_clf_ova)
coverage_ova = np.array(coverage_ova)
system_accuracy_ova = np.array(system_accuracy_ova)

accuracy_clf_ova_c = np.array(accuracy_clf_ova_c)
coverage_ova_c = np.array(coverage_ova_c)
system_accuracy_ova_c = np.array(system_accuracy_ova_c)



accuracy_clf_defer_mean = np.mean(accuracy_clf_defer, axis=0)
accuracy_clf_defer_sem = stats.sem(accuracy_clf_defer, axis=0)

system_accuracy_defer_mean = np.mean(system_accuracy_defer, axis=0)
system_accuracy_defer_sem = stats.sem(system_accuracy_defer, axis=0)

accuracy_clf_ova_mean = np.mean(accuracy_clf_ova, axis=0)
accuracy_clf_ova_sem = stats.sem(accuracy_clf_ova, axis=0)

system_accuracy_ova_mean = np.mean(system_accuracy_ova, axis=0)
system_accuracy_ova_sem = stats.sem(system_accuracy_ova, axis=0)

accuracy_clf_ova_c_mean = np.mean(accuracy_clf_ova_c, axis=0)
accuracy_clf_ova_c_sem = stats.sem(accuracy_clf_ova_c, axis=0)

system_accuracy_ova_c_mean = np.mean(system_accuracy_ova_c, axis=0)
system_accuracy_ova_c_sem = stats.sem(system_accuracy_ova_c, axis=0)

coverage_defer_mean = np.mean(coverage_defer, axis=0)
coverage_defer_sem = stats.sem(coverage_defer, axis=0)
coverage_ova_mean = np.mean(coverage_ova, axis=0)
coverage_ova_sem = stats.sem(coverage_ova, axis=0)

coverage_ova_c_mean = np.mean(coverage_ova_c, axis=0)
coverage_ova_c_sem = stats.sem(coverage_ova_c, axis=0)


# plot system accuracy and coverage
fig, axs = plt.subplots(1,2, figsize=(7.5, 5))
axs[0].plot([2,4,6,8], system_accuracy_defer_mean, marker='o', color='firebrick', markersize=4, linewidth=2, label='softmax')
axs[0].fill_between([2,4,6,8],system_accuracy_defer_mean - system_accuracy_defer_sem,  system_accuracy_defer_mean + system_accuracy_defer_sem, alpha=0.3, color='firebrick')
axs[0].plot([2,4,6,8], system_accuracy_ova_mean, marker='o', color='blue', markersize=4, linewidth=2, label='OvA')
axs[0].fill_between([2,4,6,8],system_accuracy_ova_mean - system_accuracy_ova_sem,  system_accuracy_ova_mean + system_accuracy_ova_sem, alpha=0.3, color='blue')

axs[1].plot(coverage_defer_mean, accuracy_clf_defer_mean, marker='o', color='firebrick', markersize=4, linewidth=2, label='softmax')
axs[1].fill_between(coverage_defer_mean, accuracy_clf_defer_mean - accuracy_clf_defer_sem, accuracy_clf_defer_mean + accuracy_clf_defer_sem, color='firebrick', alpha=0.3)
axs[1].plot(coverage_ova_mean, accuracy_clf_ova_mean, marker='o', color='blue', markersize=4, linewidth=2, label='OvA')
axs[1].fill_between(coverage_ova_mean, accuracy_clf_ova_mean - accuracy_clf_ova_sem, accuracy_clf_ova_mean + accuracy_clf_ova_sem, color='blue', alpha=0.3)


axs[0].legend(loc='best')
axs[0].tick_params(axis='both', which='major')
axs[0].tick_params(axis='both', which='major')
axs[0].set_xlabel(r'k (classes expert predicts correctly)')
axs[0].set_ylabel(r'System accuracy')
axs[1].set_xlabel(r'Coverage')
axs[1].set_ylabel(r'Classifier accuracy')
axs[0].grid()
axs[1].grid()
plt.tight_layout()
plt.show()
fig.savefig('./Softmax_vs_OvA_new.pdf')
#fig.savefig('./../../Figures/Softmax_vs_OvA_new.pdf')





# plot with temperature scaling (post-hoc calibration)
fig, axs = plt.subplots(figsize=(5, 5))
axs.plot([2,4,6,8], system_accuracy_defer_mean, marker='o', color='firebrick', markersize=4, linewidth=2, label='softmax')
axs.fill_between([2,4,6,8],system_accuracy_defer_mean - system_accuracy_defer_sem,  system_accuracy_defer_mean + system_accuracy_defer_sem, alpha=0.3, color='firebrick')
axs.plot([2,4,6,8], system_accuracy_ova_mean, marker='o', color='blue', markersize=4, linewidth=2, label='OvA')
axs.fill_between([2,4,6,8],system_accuracy_ova_mean - system_accuracy_ova_sem,  system_accuracy_ova_mean + system_accuracy_ova_sem, alpha=0.3, color='blue')
axs.plot([2,4,6,8], system_accuracy_ova_c_mean, marker='x', color='purple', markersize=4, linestyle='dashed', linewidth=2, label='OvA temp scaling')
axs.fill_between([2,4,6,8],system_accuracy_ova_c_mean - system_accuracy_ova_c_sem,  system_accuracy_ova_c_mean + system_accuracy_ova_c_sem, alpha=0.3, color='purple')

axs.legend(loc='best')
axs.tick_params(axis='both', which='major')
axs.tick_params(axis='both', which='major')
axs.set_xlabel(r'k (classes expert predicts correctly)')
axs.set_ylabel(r'System accuracy')
axs.grid()
plt.tight_layout()
plt.show()
fig.savefig('./Softmax_vs_OvA_with_Temp_Scaling.pdf')
#fig.savefig('./../../Figures/Softmax_vs_OvA_with_Temp_Scaling.pdf')