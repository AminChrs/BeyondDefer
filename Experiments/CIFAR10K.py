from basic import general_experiment
from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
def plot_res_vs_k(results, methods, filename, k_range):

    acc = []
    for i, res in enumerate(results):
        for k in enumerate(k_range):
            acc.append(res[k][0]["system_acc"])

        plt.plot(k_range, acc, label=methods[i])
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("system accuracy")
    plt.title("system accuracy vs k")
    filename_sp = filename.split("/")
    str_filename = ""
    if len(filename_sp) > 1:
        for i in range(len(filename_sp)-1):
            if not os.path.exists(filename_sp[i]):
                os.mkdir(str_filename + filename_sp[i])
            str_filename += filename_sp[i] + "/"
    plt.savefig(filename)


if __name__ == "__main__":

    k_range = np.arange(1, 10)
    methods = ["Beyond Defer", "Reallizable Surrogate", "Compare Confindences", "One-versus-All", "Cross Entropy"]
    res = []
    for k in k_range:
        dataset = CifarSynthDataset(k, False, batch_size=512)
        res_pack = general_experiment(dataset, "cifar_synth", 80, 10, device,
                                      subsample=False, iter=0)
        res.append(res_pack)
    res = list(res_pack)
    filename = "SampleComp/cifar_synth/acc_vs_k.png"
    plot_res_vs_k(res, methods, filename, k_range)
