from Experiments.basic import general_experiment
from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from Experiments.basic_parallel import experiment_parallel, return_res
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_res_vs_k(results, methods, filename, k_range):
    logging.info("len(results[0]): {}".format(len(results[0])))
    for i in range(len(results[0])):
        acc = []
        acc2 = []
        for j in range(len(results)):
            if j // 9 == 0:
                acc.append(results[j][i][0]["system_acc"])
                acc2.append((results[j][i][0]["system_acc"])**2)
            else:
                acc[j % 9] += results[j][i][0]["system_acc"]
                acc2[j % 9] += (results[j][i][0]["system_acc"])**2
        acc = np.array(acc)
        acc2 = np.array(acc2)
        acc /= (len(results) / 9)
        acc2 /= (len(results) / 9)
        acc_std = np.sqrt(acc2 - acc**2)
        plt.errorbar(k_range, acc, yerr=acc_std, label=methods[i], marker='o')
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("system accuracy")
    # plt.title("system accuracy vs k")
    plt.xticks(k_range)
    filename_sp = filename.split("/")
    str_filename = ""
    if len(filename_sp) > 1:
        for i in range(len(filename_sp)-1):
            if not os.path.exists(str_filename + filename_sp[i]):
                os.mkdir(str_filename + filename_sp[i])
            str_filename += filename_sp[i] + "/"
    plt.savefig(filename)


def CIFAR10K_init():
    k_range = np.arange(1, 10)
    methods = ["Additional Beyond Defer",
               "Learned Additional Beyond", "Compare Confidences Additional",
               "Reallizable Surrogate", "Compare Confindences",
               "One-versus-All", "Cross Entropy"]
    res = []
    return return_res(k_range=k_range, methods=methods, res=res)


def CIFAR10K_for_loop(res, iter):
    expert_num = iter % 9
    dataset = CifarSynthDataset(res.k_range[expert_num], False, batch_size=512)
    res_pack = general_experiment(dataset, "cifar_synth", 150, 10, device,
                                  subsample=False, iter=0)
    return return_res(res_pack=res_pack, methods=res.methods,
                      k_range=res.k_range)


def CIFAR10K_conc(cls, res):
    filename = "Results/acc_vs_k/acc_vs_k.pdf"
    accs = []
    for i, _ in enumerate(res):
        accs.append(res[i].res_pack)
    plot_res_vs_k(accs, res[0].methods, filename, res[0].k_range)


def Exp_serial():
    Parallel = experiment_parallel(CIFAR10K_for_loop,
                                   CIFAR10K_init,
                                   CIFAR10K_conc,
                                   9,
                                   "data/CIFAR10K/acc_vs_k")
    Parallel.run(parallel=False)


def Exp_parallel(iter):
    Parallel = experiment_parallel(CIFAR10K_for_loop,
                                   CIFAR10K_init,
                                   CIFAR10K_conc,
                                   90,
                                   "data/CIFAR10K/acc_vs_k")
    Parallel.run(parallel=True, iter=iter)
