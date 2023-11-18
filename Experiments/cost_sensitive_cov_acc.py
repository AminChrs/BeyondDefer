from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_deferral.datasetsdefer.hatespeech import HateSpeech
from human_ai_deferral.datasetsdefer.imagenet_16h import ImageNet16h
from human_ai_deferral.datasetsdefer.cifar_h import Cifar10h
from Experiments.basic import cov_vs_acc_add, compute_coverage_v_acc_curve
from Experiments.basic import plot_cov_vs_cost
from Experiments.basic_parallel import experiment_parallel, return_res
from metrics.metrics import aggregate_plots
from torch.utils.data import DataLoader
from MyNet.call_net import networks, optimizer_scheduler
from MyMethod.additional_cost import AdditionalCost
from baselines.lce_cost import LceCost
from baselines.compare_confidence_cost import CompareConfCost
from baselines.one_v_all_cost import OVACost
import warnings
import numpy as np
import torch
import logging
warnings.filterwarnings("ignore")
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def general_experiment_cost(dataset, dataset_name, epochs, num_classes, device,
                            loss_matrix, subsample=True, iter=0, method="cov",
                            ):

    # Data
    Dataset = dataset
    train_dataset = Dataset.data_train_loader.dataset
    len_train = len(train_dataset)
    # intervals = 10000
    # num_intervals = int(len_train / intervals)
    prop = 0.05
    intervals = int(len_train * prop)
    num_intervals = 20
    if subsample and iter < num_intervals:
        Dataset.data_train_loader = DataLoader(
            train_dataset, batch_size=512,
            sampler=torch.utils.data.SubsetRandomSampler(
                np.arange(0, (iter + 1) * intervals)))
    elif iter >= num_intervals and num_intervals != 0:
        return False
    # AB
    logging.info("Additional Beyond")
    classifier, human, meta = networks(dataset_name, "Additional", device)
    AB = AdditionalCost(10, classifier, human, meta, device, loss_matrix)
    optimizer, scheduler = optimizer_scheduler()
    AB.fit(dataset.data_train_loader, dataset.data_val_loader,
           dataset.data_test_loader, num_classes, epochs, optimizer, lr=1e-3,
           scheduler=scheduler, verbose=False)
    test_data = AB.test(dataset.data_test_loader, num_classes)
    res_AB = cov_vs_acc_add(test_data, method=method, loss_matrix=loss_matrix)

    # CC
    logging.info("Compare Confidence")
    model_class, model_expert = networks(dataset_name, "confidence", device)
    CC = CompareConfCost(model_class, model_expert, device)
    CC.set_loss_matrix(loss_matrix)
    CC.fit(
        Dataset.data_train_loader,
        Dataset.data_val_loader,
        Dataset.data_test_loader,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        lr=0.001,
        verbose=False,
        test_interval=5,
    )
    test_data = CC.test(dataset.data_test_loader)
    res_CC = compute_coverage_v_acc_curve(test_data, method=method,
                                          loss_matrix=loss_matrix)
    # LCE
    logging.info("LCE")
    model = networks(dataset_name, "LCE", device)
    LCE = LceCost(1, 2, model, device)
    LCE.set_loss_matrix(loss_matrix)
    LCE.fit_hyperparam(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=0.001,
            verbose=False,
            test_interval=5,
        )
    test_data = LCE.test(dataset.data_test_loader)
    res_LCE = compute_coverage_v_acc_curve(test_data, method=method,
                                           loss_matrix=loss_matrix)
    # OVA
    logging.info("OVA")
    model = networks(dataset_name, "OVA", device)
    OVA = OVACost(1, 2, model, device)
    OVA.set_loss_matrix(loss_matrix)
    OVA.fit(
            Dataset.data_train_loader,
            Dataset.data_val_loader,
            Dataset.data_test_loader,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=0.001,
            verbose=False,
            test_interval=5,
        )
    test_data = OVA.test(dataset.data_test_loader)
    res_OVA = compute_coverage_v_acc_curve(test_data, method=method,
                                           loss_matrix=loss_matrix)
    if method == "cov":
        return res_AB, res_CC, res_LCE, res_OVA
    else:
        return res_AB, res_CC, res_LCE, res_OVA


def acc_cov_cost_init():
    methods = ["Additional Beyond Defer", "Compare Confidences",
               "Cross Entropy", "One-vs-All"]
    datasets = ["cifar", "cifar10h", "hatespeech", "imagenet"]
    res_dir = "./Results/loss_vs_cov_cost/"
    # epochs = [150, 150, 150, 150]
    # epochs = [1, 1, 1, 1]
    epochs = [30, 30, 30, 30]
    return return_res(methods=methods, res_dir=res_dir, epochs=epochs,
                      datasets=datasets)


def acc_cov_cost_loop(res, iter):

    results = {}
    # Cifar
    dataset_cifar = CifarSynthDataset(5, False, batch_size=512)
    loss_matrix = torch.rand(10, 10)
    loss_matrix = loss_matrix - torch.diag(torch.diag(loss_matrix))
    packed_res = general_experiment_cost(dataset_cifar, "cifar_synth",
                                         res.epochs[0], 10, device,
                                         loss_matrix,
                                         subsample=False, iter=0)
    results["cifar"] = list(packed_res)

    # CIFAR10H
    dataset_cifar10h = Cifar10h(False, data_dir='./data')
    loss_matrix = torch.rand(10, 10)
    loss_matrix = loss_matrix - torch.diag(torch.diag(loss_matrix))
    packed_res = general_experiment_cost(dataset_cifar10h, "cifar_10h",
                                         res.epochs[1], 10, device,
                                         loss_matrix,
                                         subsample=False, iter=0)

    results["cifar10h"] = list(packed_res)

    # Hate Speech
    dataset_hate = HateSpeech("./data/", True, False, 'random_annotator',
                              device)
    loss_matrix = torch.rand(3, 3)
    loss_matrix = loss_matrix - torch.diag(torch.diag(loss_matrix))
    packed_res = general_experiment_cost(dataset_hate, "hatespeech",
                                         res.epochs[2], 3, device,
                                         loss_matrix,
                                         subsample=False, iter=0)
    results["hatespeech"] = list(packed_res)

    # Imagenet
    dataset_imagenet = ImageNet16h(False,
                                   data_dir="./data/osfstorage-archive/",
                                   noise_version="110",
                                   batch_size=32,
                                   test_split=0.2,
                                   val_split=0.01)
    loss_matrix = torch.rand(16, 16)
    loss_matrix = loss_matrix - torch.diag(torch.diag(loss_matrix))
    packed_res = general_experiment_cost(dataset_imagenet, "imagenet",
                                         res.epochs[3], 16, device,
                                         loss_matrix,
                                         subsample=False, iter=0)
    results["imagenet"] = list(packed_res)
    return return_res(results=results,
                      datasets=res.datasets,
                      methods=res.methods,
                      res_dir=res.res_dir)


def acc_cov_cost_conc(cls, res):
    logging.info("Concluding...")
    x_out = np.arange(0, 1, 0.01)
    for dataset in res[0].datasets:
        Res_methods = []
        for j in range(len(res[0].methods)):
            accs = []
            covs = []
            losses = []
            Res = []
            for i in range(len(res)):
                accs_i = [m["system_acc"] for m
                          in res[i].results[dataset][j]]
                covs_i = [m["coverage"] for m
                          in res[i].results[dataset][j]]
                loss_i = [m["system_loss"] for m
                          in res[i].results[dataset][j]]
                accs.append(accs_i[1:])
                covs.append(covs_i[1:])
                losses.append(loss_i[1:])
            # find maximum accuracy
            accs_o, accs_std = aggregate_plots(covs, accs, x_out,
                                               method="avg")
            loss_o, loss_std = aggregate_plots(covs, losses, x_out,
                                               method="avg")
            for i in range(len(accs_o)):
                Res.append({"system_acc": accs_o[i], "coverage": x_out[i],
                            "std_acc": accs_std[i], "system_loss": loss_o[i],
                            "std_loss": loss_std[i]})
            Res_methods.append(Res)
        filename = res[0].res_dir + dataset + ".pdf"
        plot_cov_vs_cost(Res_methods, res[0].methods, filename)


def cov_loss_parallel(iter):
    Parallel = experiment_parallel(acc_cov_cost_loop,
                                   acc_cov_cost_init,
                                   acc_cov_cost_conc,
                                   20,
                                   "data/loss_vs_cov/")
    Parallel.run(parallel=True, iter=iter)
