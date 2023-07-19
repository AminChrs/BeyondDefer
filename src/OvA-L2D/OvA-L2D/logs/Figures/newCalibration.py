from __future__ import division
import os
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

from new_reliability import *
from reliability_diagrams import _reliability_diagram_combined

def Tensorize(input):
    if not torch.is_tensor(input):
        return torch.tensor(input)
    else:
        return input

def Stat(logits, label):
    logits, label = Tensorize(logits), Tensorize(label)


    prob = F.softmax(logits, dim=1)
    conf, pred = torch.max(prob, dim=1)
    idx = torch.where(pred==10)
    label_of_defer = label[idx]
    proportion = []
        
    for c in range(10):
        total_c = sum(torch.eq(label, torch.tensor([c])))
        total_c_defer = sum(torch.eq(label_of_defer, torch.tensor([c])))
        proportion.append(total_c_defer)
    
    print(proportion)

class DirichletCalibration(nn.Module):
    def __init__(self, K):
        super(DirichletCalibration, self).__init__()
        self.K = K
        self.Linear = nn.Linear(self.K, K)

    def forward(self, probs):
        probs = torch.log(probs)
        probs = self.Linear(probs)
        probs = F.softmax(probs)
        return probs



class TempScaling(nn.Module):
    def __init__(self, K=10):
        super(TempScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1)*1)
        self.gamma = nn.Parameter(torch.randn(1)*1)
        self.K = K
    def forward(self, prob):
        prob = self.temperature*(prob - self.gamma)
        return prob


def Compare(k, seed, logits, true, expert):
    logits, true, expert = Tensorize(logits), Tensorize(true), Tensorize(expert)

    print(logits.shape, true.shape, expert.shape)

    #probs
    probs = F.sigmoid(logits)
    expert_conf = probs[:,10]
    expert_accuracies = expert.eq(true)
    expert_accuracies = expert_accuracies.float()


    log = compute_calibration(expert_conf, expert_accuracies)
    fig, _ =  _reliability_diagram_combined(log, True, 'alpha',
                                         True, "", figsize=(4,4), 
                                         dpi=72, return_fig=True)
    plt.show()
    os.makedirs('./../Figs2/', exist_ok=True)
    fig.savefig('./../Figs2/expert_reliability_uncalibrated_K_' + str(k) + '_seed_' + str(seed) + '.png')


    def do(calibrator, prob, labels):
        Criterion = nn.BCELoss()
        optimizer = optim.Adam(calibrator.parameters(), lr=1.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1)
        epoch_loss = []
        best_loss = np.inf
        for iter in range(500):
            calibrator.zero_grad()
            temp = calibrator(prob)
            loss = Criterion(F.sigmoid(temp), labels)
            # loss = loss + 0.001*torch.norm((calibrator.Linear.weight)*(1.0 - torch.eye(2)), p=2) + 0.001*torch.norm(calibrator.Linear.bias, p=2)
            loss.backward()
            scheduler.step(loss)
            epoch_loss.append(loss.item())
            optimizer.step()
            
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.show()
        for name in calibrator.named_parameters():
            print(name)
    calibrator = TempScaling()
    do(calibrator, expert_conf, expert_accuracies)
    
    os.makedirs('./../Calibrator_New', exist_ok=True)
    torch.save(calibrator.state_dict(), './../Calibrator_New/Temp_Scaling_K_' + str(k) + '_seed_' + str(seed) + '.pt')
    # calibrator.load_state_dict(torch.load('./../Calibrator/3_Dirichlet_calibrator_alpha_'+str(alpha)+'.pt'))

    #plot the reliability diagram
    p = calibrator(expert_conf)
    #np.savetxt('probs_after.txt', probs.detach(), fmt='%1.4f')
    log = compute_calibration(F.sigmoid(p), expert_accuracies)
    fig, _ =  _reliability_diagram_combined(log, True, 'alpha',
                                         True, "", figsize=(4,4), 
                                         dpi=72, return_fig=True)
    plt.show()
    fig.savefig('./../Figs2/expert_reliability_calibrated_K_' + str(k) + '_seed_' + str(seed) + '.png') 





if __name__ == "__main__":
    
    path = './../../OVA_Results_Ks_Experiment_Validation/expert_predict_'
    for seed in [948, 625, 436, 791, 1750]:

        with open(path + 'logits_alpha_seed_' + str(seed) + '.txt', 'r') as f:
            logits = json.loads(json.load(f))

        with open(path + 'true_label_seed_' + str(seed) + '.txt', 'r') as f:
            true = json.loads(json.load(f))
        
        with open(path + 'expert_predictions_seed_' + str(seed) + '.txt', 'r') as f:
            expert_pred = json.loads(json.load(f))
        
        for k in [2,4,6,8]:
            logits_k = logits[str(k)]
            true_k = true[str(k)]
            expert_pred_k = expert_pred[str(k)]

            logits_k_cifar = logits_k['cifar']
            true_k_cifar = true_k['cifar']
            expert_pred_k_cifar = expert_pred_k['cifar']

            Compare(k, seed, logits_k_cifar, true_k_cifar, expert_pred_k_cifar)
            # break
        # break


    