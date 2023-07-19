import json
import numpy as np
import torch.nn as nn
from newCalibration import Tensorize
from new_reliability import *
import seaborn as sns
from matplotlib import rc
from reliability_diagrams import _reliability_diagram_combined
plt.rcParams['xtick.labelsize'] = 42
plt.rcParams['ytick.labelsize'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# plt.rc('font', weight='bold')
# plt.style.use('seaborn')


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

sigmoid = nn.Sigmoid()

path = './Data/OVA_Results_BiasedK_Expert_NoWarmups/expert_biasedK5_'
#path = './../../OVA_Results_BiasedK_Expert_NoWarmups/expert_biasedK5_'

with open(path+'logits_alpha.txt', 'r') as f:
    data = json.loads(json.load(f))['5']

with open(path+'expert_predictions.txt', 'r') as f:
    expert = json.loads(json.load(f))['5']

with open(path+'true_label.txt', 'r') as f:
    true = json.loads(json.load(f))['5']


logits_cifar = sigmoid(Tensorize(data['cifar']))
expert_cifar = Tensorize(expert['cifar'])
true_cifar = Tensorize(true['cifar'])

conf_pred, pred = torch.max(logits_cifar, dim=1)


#reliability for the classifier
conf_clf, pred_clf = torch.max(logits_cifar[:,:10], dim=1)
acc = pred_clf.eq(true_cifar)
conf = conf_clf
accuracies = acc.float()

# reliability for the rejector
logits_expert = logits_cifar[:,10]
conf = logits_expert
acc = expert_cifar.eq(true_cifar)
accuracies = acc.float()

log = compute_calibration(conf, accuracies)
fig1, fig2 =  _reliability_diagram_combined(log, True, 'alpha',
                                        True, "", figsize=(4,4), 
                                        dpi=72, return_fig=True)
plt.show()
fig1.savefig('./R_Expert_BiasedK5_uncalibrated_Rejector_OvA_new_No_Warmups.pdf')



true_loss = torch.ones(true_cifar.shape[0])
ids_class_lt_k = torch.where(true_cifar <= 5)
true_loss[ids_class_lt_k] = 1.0 - 0.75
ids_class_gt_k = torch.where(true_cifar > 5)
true_loss[ids_class_gt_k] = 1.0 - 0.1


defer_prob = logits_expert
predicted_loss = 1.0-defer_prob

#plot true loss and predicted loss
fig, ax = plt.subplots(figsize=(4,4))
#plt.grid(axis='y')
ax.hist(predicted_loss.tolist(), 20, density=True, alpha=0.45, label='predicted error')
ax.hist(true_loss.tolist(), 5, density=True, alpha=0.45, facecolor='green', label='true error')
ax.legend(loc='best') #, prop={'size':12, 'weight':'bold'})
ax.set_ylabel(r"Density")
ax.set_xlabel(r"Probability of error")
#ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlim(0.0, 1.0)
ax.set_axisbelow(True)
plt.grid()
plt.tight_layout()
fig.savefig("./Predicted_Error_and_True_Error_OvA.pdf")
plt.show()