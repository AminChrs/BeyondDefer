# from human_ai_defer.datasetsdefer.cifar_synth import CifarSynthDataset
# from human_ai_defer.datasetsdefer.hatespeech import HateSpeech
# from human_ai_defer.datasetsdefer.imagenet_16h import ImageNet16h
from beyonddefer.human_ai_defer.datasetsdefer.cifar_h import Cifar10h
from beyonddefer.Experiments.basic import plot_cov_vs_acc
from beyonddefer.Experiments.basic import active_experiment
from beyonddefer.Experiments.basic_parallel import experiment_parallel, return_res
from beyonddefer.metrics.metrics import aggregate_plots
import warnings
import numpy as np
import torch
import logging
warnings.filterwarnings("ignore")
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def acc_cov_init():
    methods = ["Additional Beyond Defer",
               "Learn Additional Defer", "Compare Confidences Additional",
               "Reallizable Surrogate", "Compare Confindences",
               "One-versus-All", "Cross Entropy"]
    datasets = ["cifar", "cifar10h", "hatespeech", "imagenet"]
    res_dir = "./Results/loss_vs_cov/"
    epochs = [30, 30, 30, 30]
    # epochs = [1, 1, 1, 1]
    return return_res(methods=methods, res_dir=res_dir, epochs=epochs,
                      datasets=datasets)


def acc_cov_loop(res, iter):

    results = {}
    # # Cifar
    # dataset_cifar = CifarSynthDataset(5, False, batch_size=512)
    # packed_res = general_experiment(dataset_cifar, "cifar_synth",
    #                                 res.epochs[0], 10, device,
    #                                 subsample=False, iter=0)
    # res_active = active_experiment(dataset_cifar, "cifar_synth",
    #                                res.epochs[0], 10, 1,
    #                                device=device,)
    # packed_res = packed_res + (res_active,)
    # results["cifar"] = list(packed_res)

    # CIFAR10H
    dataset_cifar10h = Cifar10h(False, data_dir='./data')
    # packed_res = general_experiment(dataset_cifar10h, "cifar_10h",
    #                                 res.epochs[1], 10, device,
    #                                 subsample=False, iter=0)
    res_active = active_experiment(dataset_cifar10h, "cifar_10h",
                                   res.epochs[1], 10, 1,
                                   device=device,)
    packed_res = (res_active,)  # packed_res + (res_active,)
    results["cifar10h"] = list(packed_res)

    # Hate Speech
    # dataset_hate = HateSpeech("./data/", True, False, 'random_annotator',
    #                           device)
    # packed_res = general_experiment(dataset_hate, "hatespeech",
    #                                 res.epochs[2], 3, device,
    #                                 subsample=False, iter=0)
    # res_active = active_experiment(dataset_hate, "hatespeech",
    #                                res.epochs[2], 3, 1,
    #                                device=device,)
    # packed_res = packed_res + (res_active,)
    # results["hatespeech"] = list(packed_res)

    # # Imagenet
    # dataset_imagenet = ImageNet16h(False,
    #                                data_dir="./data/osfstorage-archive/",
    #                                noise_version="110",
    #                                batch_size=32,
    #                                test_split=0.2,
    #                                val_split=0.01)
    # packed_res = general_experiment(dataset_imagenet, "imagenet",
    #                                 res.epochs[3], 16, device,
    #                                 subsample=False, iter=0)
    # res_active = active_experiment(dataset_imagenet, "imagenet",
    #                                res.epochs[3], 16, 1,
    #                                device=device,)
    # packed_res = packed_res + (res_active,)
    # results["imagenet"] = list(packed_res)
    return return_res(results=results,
                      datasets=res.datasets,
                      methods=res.methods,
                      res_dir=res.res_dir)


def acc_cov_conc(cls, res):
    logging.info("Concluding...")
    x_out = np.arange(0, 1, 0.01)
    for dataset in res[0].datasets:
        Res_methods = []
        for j in range(len(res[0].methods)):
            logging.info("Method: {}".format(res[0].methods[j]))
            logging.info("Dataset: {}".format(dataset))
            accs = []
            covs = []
            Res = []
            for i in range(len(res)):
                logging.info("Iter: {}".format(i))
                accs_i = [m["system_acc"] for m
                          in res[i].results[dataset][j]]
                covs_i = [m["coverage"] for m
                          in res[i].results[dataset][j]]
                accs.append(accs_i[1:])
                covs.append(covs_i[1:])
            # find maximum accuracy
            logging.info("Dataset: {}, Method: {}".format(dataset,
                                                          res[0].methods[j]))
            accs_o, accs_std = aggregate_plots(covs, accs, x_out, method="avg")
            max_acc = np.max(accs_o)
            # round to 3 decimals
            max_acc = np.round(max_acc, 3)
            for i in range(len(accs_o)):
                Res.append({"system_acc": accs_o[i], "coverage": x_out[i],
                            "std_acc": accs_std[i]})
            Res_methods.append(Res)
        filename = res[0].res_dir + dataset + ".pdf"
        plot_cov_vs_acc(Res_methods, res[0].methods, filename)


def cov_acc_parallel(iter):
    Parallel = experiment_parallel(acc_cov_loop,
                                   acc_cov_init,
                                   acc_cov_conc,
                                   10,
                                   "data/acc_vs_cov/cifar10h")
    Parallel.run(parallel=True, iter=iter)
