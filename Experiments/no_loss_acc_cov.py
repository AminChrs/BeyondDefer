import sys
sys.path.append("../")
from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_deferral.datasetsdefer.hatespeech import HateSpeech
from human_ai_deferral.datasetsdefer.imagenet_16h import ImageNet16h
from human_ai_deferral.datasetsdefer.cifar_h import Cifar10h
from Experiments.basic import general_experiment, plot_cov_vs_acc
import warnings
import torch
warnings.filterwarnings("ignore")
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # methods
    methods = ["Beyond Defer", "Reallizable Surrogate", "Compare Confindences", "One-versus-All", "Cross Entropy"]
    res_dir = "SampleComp/"

    # Cifar
    dataset_cifar = CifarSynthDataset(5, False, batch_size=512)
    packed_res = general_experiment(dataset_cifar, "cifar_synth",
                                    80, 10, device, subsample=True, iter=0)
    results = list(packed_res)
    filename = res_dir + "cifar.pdf"
    plot_cov_vs_acc(results, methods, filename)

    # CIFAR10H
    dataset_cifar10h = Cifar10h(False, data_dir='../data')
    packed_res = general_experiment(dataset_cifar10h, "cifar_10h",
                                    80, 10, device, subsample=True, iter=0)

    results = list(packed_res)
    filename = res_dir + "cifar10h.pdf"
    plot_cov_vs_acc(results, methods, filename)

    # Hate Speech
    dataset_hate = HateSpeech("../data/", True, False, 'random_annotator',
                              device)
    packed_res = general_experiment(dataset_hate, "hatespeech",
                                    200, 3, device, subsample=True, iter=0)
    results = list(packed_res)
    filename = res_dir + "hate_speech.pdf"
    plot_cov_vs_acc(results, methods, filename)

    # Imagenet
    dataset_imagenet = ImageNet16h(False,
                                   data_dir="../data/osfstorage-archive/",
                                   noise_version="110",
                                   batch_size=32,
                                   test_split=0.2,
                                   val_split=0.01)
    packed_res = general_experiment(dataset_imagenet, "imagenet",
                                    80, 16, device, subsample=True, iter=0)
    results = list(packed_res)
    filename = res_dir + "imagenet.pdf"
    plot_cov_vs_acc(results, methods, filename)
