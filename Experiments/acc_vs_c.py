from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_deferral.datasetsdefer.hatespeech import HateSpeech
from human_ai_deferral.datasetsdefer.imagenet_16h import ImageNet16h
from human_ai_deferral.datasetsdefer.cifar_h import Cifar10h
from Experiments.basic import general_experiment, plot_cov_vs_acc
from Experiments.basic_parallel import experiment_parallel, return_res
from metrics.metrics import aggregate_plots
import warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def acc_cov_init():
    methods = ["Beyond Defer", "Additional Beyond Defer",
               "Reallizable Surrogate", "Compare Confindences",
               "One-versus-All", "Cross Entropy"]
    datasets = ["cifar", "cifar10h", "hatespeech", "imagenet"]
    res_dir = "./Results/loss_vs_cov/"
    epochs = [1, 1, 1, 1]  # [150, 150, 150, 150]
    return return_res(methods=methods, res_dir=res_dir, epochs=epochs,
                      datasets=datasets)


def acc_cov_loop(res, iter):

    results = {}
    # Cifar
    dataset_cifar = CifarSynthDataset(5, False, batch_size=512)
    packed_res = general_experiment(dataset_cifar, "cifar_synth",
                                    res.epochs[0], 10, device,
                                    subsample=False, iter=0,
                                    method="c")
    results["cifar"] = list(packed_res)

    # CIFAR10H
    dataset_cifar10h = Cifar10h(False, data_dir='./data')
    packed_res = general_experiment(dataset_cifar10h, "cifar_10h",
                                    res.epochs[1], 10, device,
                                    subsample=False, iter=0,
                                    method="c")

    results["cifar10h"] = list(packed_res)

    # Hate Speech
    dataset_hate = HateSpeech("./data/", True, False, 'random_annotator',
                              device)
    packed_res = general_experiment(dataset_hate, "hatespeech",
                                    res.epochs[2], 3, device,
                                    subsample=False, iter=0,
                                    method="c")
    results["hatespeech"] = list(packed_res)

    # Imagenet
    dataset_imagenet = ImageNet16h(False,
                                   data_dir="./data/osfstorage-archive/",
                                   noise_version="110",
                                   batch_size=32,
                                   test_split=0.2,
                                   val_split=0.01)
    packed_res = general_experiment(dataset_imagenet, "imagenet",
                                    res.epochs[3], 16, device,
                                    subsample=False, iter=0,
                                    method="c")
    results["imagenet"] = list(packed_res)
    return return_res(results=results,
                      datasets=res.datasets,
                      methods=res.methods,
                      res_dir=res.res_dir)


def acc_cov_conc(cls, res):
    x_out = np.arange(0, 1, 0.01)
    for dataset in res[0].datasets:
        Res_methods = []
        for j in range(len(res[0].methods)):
            accs = []
            covs = []
            Res = []
            for i in range(len(res)):
                accs_i = [m["system_acc"] for m
                          in res[i].results[dataset][j]]
                covs_i = [m["coverage"] for m
                          in res[i].results[dataset][j]]
                accs.append(accs_i[1:])
                covs.append(covs_i[1:])
            accs_o = aggregate_plots(covs, accs, x_out, method="avg")
            for i in range(len(accs_o)):
                Res.append({"system_acc": accs_o[i], "coverage": x_out[i]})
            Res_methods.append(Res)
        filename = res[0].res_dir + dataset + ".pdf"
        plot_cov_vs_acc(Res_methods, res[0].methods, filename)


def acc_c_parallel(iter):
    Parallel = experiment_parallel(acc_cov_loop,
                                   acc_cov_init,
                                   acc_cov_conc,
                                   10,
                                   "data/acc_vs_c/")
    Parallel.run(parallel=False, iter=iter)
