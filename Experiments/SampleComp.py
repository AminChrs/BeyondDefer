# In this file, I compare all the methods in terms of sample complexity
# for 4 datasets CIFAR 10k, CIFAR 10H, Imagenet, and Hatespeech

import sys
sys.path.append("../")
from Feature_Acquisition.active import IndexedDataset, ActiveDataset, AFE
from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_deferral.datasetsdefer.cifar_h import Cifar10h
from human_ai_deferral.datasetsdefer.hatespeech import HateSpeech
from human_ai_deferral.datasetsdefer.imagenet_16h import ImageNet16h
from MyNet.networks import MetaNet
from human_ai_deferral.networks.cnn import WideResNet
from human_ai_deferral.baselines.selective_prediction import\
     SelectivePrediction
from MyMethod.beyond_defer import BeyondDefer
from human_ai_deferral.networks.cnn import NetSimple
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
import logging
import matplotlib.pyplot as plt
import torch.nn as nn
import os


def optimizer_scheduler():

    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       weight_decay=0.0005)
    return optimizer, scheduler


def datasets(name, num_train, device):

    if name[:11] == "cifar_synth":
        expert_k = int(name[12:])
        dataset = CifarSynthDataset(expert_k, False, batch_size=512)
    elif name == "cifar_10h":
        dataset = Cifar10h(False, data_dir='../data')
    elif name == "hatespeech":
        dataset = HateSpeech("../data", True, False, 'random_annotator',
                             device)
    elif name == "imagenet":
        dataset = ImageNet16h(False, data_dir="../data" +
                              "/osfstorage-archive/", noise_version=125,
                              batch_size=32, test_split=0.2, val_split=0.01)

    dataset_train = dataset.data_train_loader.dataset
    dataset.data_train_loader = DataLoader(
        dataset=dataset_train, batch_size=512, shuffle=True,
        sampler=torch.utils.data.SubsetRandomSampler(
            np.arange(num_train)))
    return dataset


def main():

    datasets = ["cifar_synth_10", "cifar_10h", "hatespeech", "imagenet"]
    methods = ["BD", "AFE", "triage", "confidence", "selective"]
    