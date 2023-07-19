from __future__ import division
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
from newCalibration import Tensorize
import seaborn as sns
from matplotlib import rc
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


softmax = nn.Softmax(dim=1)
severity_levels = ['cifar']


path = './../Data/Sontag_Results_BiasedK_Expert/expert_biasedK5_'
#path = './../../Sontag_Results_BiasedK_Expert/expert_biasedK5_'          #Sontag_Results_Random_Expert/expert_random_'




def func(i):
    return 1 - i

def read(path):
    confs = []
    coverage = []

    for k in [5]:
        with open(path+'logits_alpha_1.0.txt', 'r') as f:
            data = json.loads(json.load(f))[str(k)]

        conf = {}
        for severity in severity_levels:
            probs = softmax(Tensorize(data[severity]))
            c, m = torch.max(probs, dim=1)
            reject_idx = torch.where(m == 10)
            temp = torch.sum(probs[:,:10], dim=1, keepdim=True)
            probs = probs/temp
            conf[severity] = probs[reject_idx][:,10]

        columns = [conf[severity] for severity in severity_levels]
        confs.append(columns)

        with open(path + 'validation_alpha_1.0.txt', 'r') as f:
            val = json.load(f)[str(k)]
        
        cov_ = {}
        for severity in severity_levels:
            coverage_ = val[severity]["coverage"]
            cov, out_of = coverage_.split(' ')[0], coverage_.split('of')[-1]
            cov_[severity] = int(cov) / int(out_of)
        
        columns = [cov_[severity] for severity in severity_levels]
        vals = list(map(func, columns))
        coverage.append(vals)
    return confs, coverage



confs, coverage = read(path)

total = 0
for c in confs[0][0]:
    if c>1.0:
        total+=1
    
#print(total, len(confs[0][0]))

import matplotlib
#plt.rcParams.update({"text.usetex": True})
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{bm}"]
plt.rcParams['hatch.linewidth'] = 3
from matplotlib.ticker import FormatStrFormatter
fig, ax = plt.subplots(figsize=(4.5,4))
N, bins, patches = ax.hist(confs[0][0].tolist(), bins=10, alpha=0.5)
print(type(patches), len(patches))
for i in range(0,2):
    patches[i].set_alpha(0.325)
    patches[i].set_linewidth(2.0)
for i in range(2, len(patches)):
    patches[i].set_facecolor('red')
    patches[i].set_alpha(0.60)
    patches[i].set_hatch('//')
    patches[i].set_edgecolor('firebrick')
    #patches[i].set_linewidth(2.0)
ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='both', which='major')
ax.set_xlabel(r"$p_{\boldmath{m}}$(x)")
ax.set_ylabel(r"Count")
ax.axvline(x=1.0325, color='firebrick', linestyle='dashed', linewidth=5)
ax.set_ylim(0, 400)
ax.grid()
plt.tight_layout()
fig.savefig('./histogram.pdf')
#fig.savefig("./../../Figures/histogram.pdf")
plt.show()


