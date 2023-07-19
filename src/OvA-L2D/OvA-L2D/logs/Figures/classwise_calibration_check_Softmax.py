from __future__ import division
import json
import numpy as np
from scipy import stats
import torch.nn as nn
from newCalibration import Tensorize
from new_reliability import *
from reliability_diagrams import _reliability_diagram_combined
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

#plt.style.use('seaborn')

softmax = nn.Softmax(dim=1)


path = './../Data/Sontag_Results_BiasedK_Expert/expert_biasedK5_'
#path = './../../Sontag_Results_BiasedK_Expert/expert_biasedK5_'


with open(path+'logits_alpha_1.0.txt', 'r') as f:
    data = json.loads(json.load(f))['5']

with open(path+'expert_predictions_alpha_1.0.txt', 'r') as f:
    expert = json.loads(json.load(f))['5']

with open(path+'true_label_alpha_1.0.txt', 'r') as f:
    true = json.loads(json.load(f))['5']


logits_cifar = softmax(Tensorize(data['cifar']))
temp = torch.sum(logits_cifar[:,:10], dim=1, keepdim=True)
logits_cifar_ = logits_cifar/temp
expert_cifar = Tensorize(expert['cifar'])
true_cifar = Tensorize(true['cifar'])

conf_pred, pred = torch.max(logits_cifar, dim=1)


#reliability for the classifier
conf_clf, pred_clf = torch.max(logits_cifar_[:,:10], dim=1)
acc = pred_clf.eq(true_cifar)
conf = conf_clf
accuracies = acc.float()
log = compute_calibration(conf, accuracies)


#reliability for rejector
logits_expert = logits_cifar_[:,10]
ids_where_gt_one = torch.where(logits_expert > 1.0)
logits_expert[ids_where_gt_one] = 1.0
conf = logits_expert
acc = expert_cifar.eq(true_cifar)
accuracies = acc.float()

log = compute_calibration(conf, accuracies)
fig1, fig2 =  _reliability_diagram_combined(log, True, 'alpha',
                                        False, "", figsize=(4,4), 
                                        dpi=72, return_fig=True)
fig1.savefig('./R_Expert_BiasedK5_uncalibrated_Rejector_Sontag_new.pdf')
#fig1.savefig('./../../Figures/R_Expert_BiasedK5_uncalibrated_Rejector_Sontag_new.pdf')

true_loss = torch.ones(true_cifar.shape[0])
ids_class_lt_k = torch.where(true_cifar <= 5)
true_loss[ids_class_lt_k] = 1.0 - 0.75
ids_class_gt_k = torch.where(true_cifar > 5)
true_loss[ids_class_gt_k] = 1.0 - 0.1

print(true_loss.shape, len(ids_class_lt_k[0]))

defer_prob = logits_expert
predicted_loss = 1.0-defer_prob

#plot true loss and predicted loss
plt.rcParams['hatch.linewidth'] = 3
fig, ax = plt.subplots(figsize=(4,4))
N, bins, patches = ax.hist(predicted_loss.tolist(), 20, density=True, alpha=0.45, label='predicted error')
ax.hist(true_loss.tolist(), 5, density=True, alpha=0.45, facecolor='green', label='true error')
ax.legend(loc='best')
for i in range(0, 1):
    #patches[i].set_facecolor('blue')
    patches[i].set_alpha(0.60)
    patches[i].set_hatch('//')
    patches[i].set_edgecolor('blue')
    #patches[i].set_linewidth(2.0)

ax.annotate('not faithful', xy=(0.02, 1.5),
            xytext = (0.012, 2.55), fontsize=11,
            arrowprops=dict(facecolor='black', shrink=0.005),
            )
ax.set_ylabel(r"Density")
ax.set_xlabel(r"Probability of error")
ax.tick_params(axis='both', which='major')
ax.grid()
plt.tight_layout()
plt.show()
fig.savefig("./Predicted_Error_and_True_Error_Sontag.pdf")
