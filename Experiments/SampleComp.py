from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_deferral.datasetsdefer.hatespeech import HateSpeech
from human_ai_deferral.datasetsdefer.imagenet_16h import ImageNet16h
from human_ai_deferral.datasetsdefer.cifar_h import Cifar10h
from Experiments.basic import general_experiment, active_experiment
import matplotlib.pyplot as plt
import torch
import warnings
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

    accs = []
    plt.figure()

    for i in range(len(methods)):
        accs.append([results[:][i]])
        plt.plot(samples, accs[i], label=methods[i])

    plt.xlabel("samples")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    plt.savefig(filename)

if __name__ == "Experiments.SampleComp":
    # methods
    methods = ["Beyond Defer", "Reallizable Surrogate", "Compare Confindences",
               "One-versus-All", "Cross Entropy"]
    res_dir = "SampleComp/"

    dataset_cifar = CifarSynthDataset(5, False, batch_size=512)
    dataset_cifar10h = Cifar10h(False, data_dir='./data')
    dataset_hate = HateSpeech("../data/", True, False, 'random_annotator',
                              device)
    # print the current folder
    dataset_imagenet = ImageNet16h(False,
                                   data_dir="./data/osfstorage-archive/",
                                   noise_version="110", batch_size=32,
                                   test_split=0.2,
                                   val_split=0.01)

    datasets = [dataset_cifar, dataset_cifar10h, dataset_hate,
                dataset_imagenet]
    i = 0
    break_flag = False
    # epochs = [80, 80, 200, 80]
    epochs = [1, 1, 1, 1]
    num_classes = [10, 10, 3, 16]
    names = ["cifar_synth", "cifar_10h", "hatespeech", "imagenet"]
    accs = {}
    for i, name in enumerate(names):
        accs[name] = []

    while (True):

        for j, name in enumerate(names):

            packed_res = general_experiment(datasets[j], names[j],
                                            epochs[j], num_classes[j],
                                            device, subsample=True,
                                            iter=i)
            if not packed_res:
                break_flag = True
                break
            results = list(packed_res)
            accs[name].append(accuracies(results))

        if break_flag:
            break
        i += 1

    # plot
