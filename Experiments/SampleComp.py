from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_deferral.datasetsdefer.hatespeech import HateSpeech
from human_ai_deferral.datasetsdefer.imagenet_16h import ImageNet16h
from human_ai_deferral.datasetsdefer.cifar_h import Cifar10h
from Experiments.basic import general_experiment  # , active_experiment
from Experiments.basic_parallel import experiment_parallel, return_res
import matplotlib.pyplot as plt
import torch
import warnings
import logging
import numpy as np
import os
warnings.filterwarnings("ignore")
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracies(results):

    acc = []
    for i, res in enumerate(results):
        acc.append(res[0]["system_acc"])
    return acc


def plot_sample(samples, results, methods, filename):

    plt.figure()

    for i in range(len(methods)):
        acc = []
        acc_std = []
        for j in range(len(samples) // 10):
            acc_avg = 0
            acc_std = 0
            for k in range(10):
                acc_avg += results[10*j+k][0][i]
                acc_std += (results[10*j+k][0][i])**2
            acc_avg /= 10
            acc_std = np.sqrt(acc_std/10 - acc_avg**2)
            acc.append(acc_avg)
            acc_std.append(acc_std)
        plt.errorbar(samples, acc, yerr=acc_std, label=methods[i])

    plt.xlabel("samples")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    str_filename = ""
    filename_sp = filename.split("/")
    if len(filename_sp) > 1:
        for i in range(len(filename_sp)-1):
            if not os.path.exists(str_filename + filename_sp[i]):
                os.mkdir(str_filename + filename_sp[i])
            str_filename += filename_sp[i] + "/"
    plt.savefig(filename)


def SampleComp_init():
    # methods
    methods = ["Beyond Defer", "Additional Defer",
               "Reallizable Surrogate", "Compare Confindences",
               "One-versus-All", "Cross Entropy"]
    res_dir = "Results/SampleComp/"

    dataset_cifar = CifarSynthDataset(5, False, batch_size=512)
    dataset_cifar10h = Cifar10h(False, data_dir='./data')
    dataset_hate = HateSpeech("./data/", True, False, 'random_annotator',
                              device)
    # print the current folder
    dataset_imagenet = ImageNet16h(False,
                                   data_dir="./data/osfstorage-archive/",
                                   noise_version="110", batch_size=32,
                                   test_split=0.2,
                                   val_split=0.01)

    datasets = [dataset_cifar, dataset_cifar10h, dataset_hate,
                dataset_imagenet]
    break_flag = False
    epochs = [1, 1, 1, 1]  # [150, 150, 150, 150]
    num_classes = [10, 10, 3, 16]
    num_samples_per_iter = 10000
    names = ["cifar_synth", "cifar_10h", "hatespeech", "imagenet"]
    accs = {}
    for i, name in enumerate(names):
        accs[name] = []
    res = return_res(methods=methods, datasets=datasets, names=names,
                     epochs=epochs, num_classes=num_classes,
                     num_samples_per_iter=num_samples_per_iter,
                     device=device, break_flag=break_flag, res_dir=res_dir,
                     accs=accs)
    return res


def for_loop(res, iter):
    logging.info("iter = {}".format(iter))
    for j, name in enumerate(res.names):
        packed_res = general_experiment(res.datasets[j], res.names[j],
                                        res.epochs[j], res.num_classes[j],
                                        device, subsample=True,
                                        iter=iter % 10)
        if not packed_res:
            logging.info("End of the possible samples")
            res.break_flag = True
            break
        results = list(packed_res)
        res.accs[name].append(accuracies(results))
    res = return_res(accs=res.accs,
                     num_samples_per_iter=res.num_samples_per_iter,
                     res_dir=res.res_dir, methods=res.methods, names=res.names)
    return res


def SampleComp_conc(cls, res):
    sample_range = np.arange(1, (cls.iter // 10)+1)*res[0].num_samples_per_iter

    for j, name in enumerate(res[0].names):
        savedir = res[0].res_dir+"acc_vs_samples"+name+".pdf"
        accs = []
        for i in range(cls.iter):
            logging.info("accs: {}".format(res[i].accs[name]))
            accs.append(res[i].accs[name])
        plot_sample(sample_range, accs, res[0].methods, savedir)


def SampleComp_par(iter):

    # parallel
    Parallel = experiment_parallel(for_loop, SampleComp_init, SampleComp_conc,
                                   190, "data/SampleComp/parallel")
    Parallel.run(parallel=True, iter=iter)
