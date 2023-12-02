from beyonddefer.Feature_Acquisition.active import ActiveDataset, AFE
# from MyMethod.beyond_defer import BeyondDefer
from beyonddefer.MyMethod.additional_defer import AdditionalBeyond
# from MyMethod.learned_beyond import LearnedBeyond
from beyonddefer.MyMethod.learned_additional import LearnedAdditional
from beyonddefer.MyMethod.CompareConfMeta import CompareConfMeta
from beyonddefer.human_ai_defer.methods.realizable_surrogate import RealizableSurrogate
from beyonddefer.human_ai_defer.baselines.compare_confidence import CompareConfidence
from beyonddefer.human_ai_defer.baselines.lce_surrogate import LceSurrogate
from beyonddefer.human_ai_defer.baselines.one_v_all import OVASurrogate
from beyonddefer.human_ai_defer.helpers.metrics import compute_coverage_v_acc_curve
import os
# from metrics.metrics import cov_vs_acc_meta
from tikzplotlib import save as tikz_save
from metrics.metrics import cov_vs_acc_add, cov_vs_acc_AFE
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from MyNet.call_net import networks, optimizer_scheduler
import warnings
import logging
warnings.filterwarnings("ignore")


def general_experiment(dataset, dataset_name, epochs, num_classes, device,
                       subsample=True, iter=0, method="cov"):

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

    # BD
    # logging.info("Beyond Defer")
    # classifier, human, meta = networks(dataset_name, "BD", device)
    # BD = BeyondDefer(10, classifier, human, meta, device)
    # optimizer, scheduler = optimizer_scheduler()
    # BD.fit(dataset.data_train_loader, dataset.data_val_loader,
    #        dataset.data_test_loader, num_classes, epochs, optimizer, lr=1e-3,
    #        scheduler=scheduler, verbose=False)
    # test_data = BD.test(dataset.data_test_loader, num_classes)
    # res_BD = cov_vs_acc_meta(test_data, method=method)

    # AB
    logging.info("Additional Beyond")
    classifier, human, meta = networks(dataset_name, "Additional", device)
    AB = AdditionalBeyond(10, classifier, human, meta, device)
    optimizer, scheduler = optimizer_scheduler()
    AB.fit(dataset.data_train_loader, dataset.data_val_loader,
           dataset.data_test_loader, num_classes, epochs, optimizer, lr=1e-3,
           scheduler=scheduler, verbose=False)
    test_data = AB.test(dataset.data_test_loader, num_classes)
    res_AB = cov_vs_acc_add(test_data, method=method)

    # LB
    # logging.info("Learned Beyond")
    # classifier, meta = networks(dataset_name, "LearnedBeyond", device)
    # LB = LearnedBeyond(10, classifier, meta, device)
    # optimizer, scheduler = optimizer_scheduler()
    # LB.fit(dataset.data_train_loader, dataset.data_val_loader,
    #        dataset.data_test_loader, num_classes, epochs, optimizer, lr=1e-3,
    #        scheduler=scheduler, verbose=False)
    # test_data = LB.test(dataset.data_test_loader, num_classes)
    # res_LB = cov_vs_acc_meta(test_data, method=method)

    # AB
    classifier, meta = networks(dataset_name, "LearnedAdditional", device)
    LA = LearnedAdditional(10, classifier, meta, device)
    optimizer, scheduler = optimizer_scheduler()
    LA.fit(dataset.data_train_loader, dataset.data_val_loader,
           dataset.data_test_loader, num_classes, epochs, optimizer, lr=1e-3,
           scheduler=scheduler, verbose=False)
    test_data = LA.test(dataset.data_test_loader, num_classes)
    res_LA = cov_vs_acc_add(test_data, method=method)

    # CompConfMeta
    logging.info("Compare Confidence Meta")
    classifier, meta, defer, defer_meta = networks(dataset_name,
                                                   "CompConfMeta", device)
    CCM = CompareConfMeta(10, classifier, meta, defer, defer_meta, device)
    optimizer, scheduler = optimizer_scheduler()
    CCM.fit(dataset.data_train_loader, dataset.data_val_loader,
            dataset.data_test_loader, num_classes, epochs, optimizer, lr=1e-3,
            scheduler=scheduler, verbose=False)
    test_data = CCM.test(dataset.data_test_loader, num_classes)
    res_CCM = cov_vs_acc_add(test_data, method=method)

    # RS
    if method == "cov":
        logging.info("Realizable Surrogate")
        model = networks(dataset_name, "RS", device)
        Reallizable_Surr = RealizableSurrogate(1, 2, model, device, True)
        Reallizable_Surr.fit_hyperparam(
                                        Dataset.data_train_loader,
                                        Dataset.data_val_loader,
                                        Dataset.data_test_loader,
                                        epochs=epochs,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        lr=1e-3,
                                        verbose=False,
                                        test_interval=1,
                                    )
        test_data = Reallizable_Surr.test(dataset.data_test_loader)
        res_RS = compute_coverage_v_acc_curve(test_data, method=method)

    # CC
    logging.info("Compare Confidence")
    model_class, model_expert = networks(dataset_name, "confidence", device)
    CC = CompareConfidence(model_class, model_expert, device)
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
    res_CC = compute_coverage_v_acc_curve(test_data, method=method)

    # OVA
    logging.info("One vs All")
    model = networks(dataset_name, "OVA", device)
    OVA = OVASurrogate(1, 2, model, device)
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
    res_OVA = compute_coverage_v_acc_curve(test_data, method=method)

    # LCE
    logging.info("LCE")
    model = networks(dataset_name, "LCE", device)
    LCE = LceSurrogate(1, 2, model, device)
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
    res_LCE = compute_coverage_v_acc_curve(test_data, method=method)
    if method == "cov":
        return res_AB, res_LA, res_CCM, res_RS, res_CC, res_OVA, res_LCE
    else:
        return res_AB, res_LA, res_CCM, res_CC, res_OVA, res_LCE


def active_experiment(dataset, dataset_name, epochs, num_classes, num_queries,
                      len_queries=None, device="cpu"):

    Dataset = ActiveDataset(dataset)
    if len_queries is None:
        len_queries = len(Dataset.data_train_loader.dataset)
    classifier, meta = networks(dataset_name, "AFE", device)
    ActiveElicit = AFE(classifier, meta, device)
    optimizer, scheduler = optimizer_scheduler()
    ActiveElicit.fit(Dataset,
                     num_classes, epochs, lr=0.001, verbose=True,
                     query_size=len_queries,
                     num_queries=num_queries, scheduler_classifier=scheduler,
                     scheduler_meta=scheduler, optimizer=optimizer)
    last_iter = len(ActiveElicit.report) - 1
    logging.info("Last iteration: {}".format(last_iter))
    logging.info("num queries: {}".format(num_queries))
    rep = ActiveElicit.report[last_iter]
    res_AFE = cov_vs_acc_AFE(rep)
    return res_AFE


def plot_cov_vs_acc(results, methods, filename,
                    method="cov", is_loss=False, std=True):
    fig = plt.figure()
    # Get the default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i, res in enumerate(results):
        if method == "cov":
            cov = [m["coverage"] for m in res]
            acc = [m["system_acc"] for m in res]
            if std:
                acc_std = [m["std_acc"] for m in res]
            else:
                acc_std = [0 for m in res]
            # make sure that error bar is shown eacch 10 points

            plt.plot(cov[1:], acc[1:], color=colors[i])
            plt.errorbar(cov[1::10], acc[1::10], yerr=acc_std[1::10],
                         label=methods[i], fmt='o', color=colors[i])
        elif method == "c":
            c = [m["c"] for m in res]
            if not is_loss:
                acc = [m["system_acc"] for m in res]
                if std:
                    acc_std = [m["std_acc"] for m in res]
                else:
                    acc_std = [0 for m in res]
                plt.plot(c[1:], acc[1:], color=colors[i])
                plt.errorbar(c[1::10], acc[1::10], yerr=acc_std[1::10],
                             label=methods[i], fmt='o', color=colors[i])
            else:
                loss = [m["loss"] for m in res]
                if std:
                    acc_std = [m["std_acc"] for m in res]
                else:
                    acc_std = [0 for m in res]
                plt.plot(c[1:], loss[1:], color=colors[i])
                plt.errorbar(c[1::10], loss[1::10], yerr=acc_std[1::10],
                             label=methods[i], fmt='o', color=colors[i])
    plt.legend()
    if method == "cov":
        plt.xlabel("coverage")
    elif method == "c":
        plt.xlabel("deferral cost")
    if not is_loss:
        plt.ylabel("system accuracy")
    else:
        plt.ylabel("system loss")
    plt.show()

    filename_sp = filename.split("/")
    str_filename = ""
    if len(filename_sp) > 1:
        for i in range(len(filename_sp)-1):
            if not os.path.exists(str_filename+filename_sp[i]):
                logging.info("Creating directory: {}".format(filename_sp[i]))
                os.mkdir(str_filename + filename_sp[i])
            str_filename += filename_sp[i] + "/"

    plt.savefig(filename)
    # replace pdf with tex
    filename = filename.replace(".pdf", ".tex")
    tikz_save(filename, figure=fig,
              axis_width='\\textwidth', axis_height='\\textwidth')


def plot_cov_vs_cost(results, methods, filename,
                     method="cov", std=True):
    fig = plt.figure()
    # Get the default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i, res in enumerate(results):
        if method == "cov":
            cov = [m["coverage"] for m in res]
            acc = [m["system_loss"] for m in res]
            if std:
                acc_std = [m["std_loss"] for m in res]
            else:
                acc_std = [0 for m in res]
            # make sure that error bar is shown eacch 10 points

            plt.plot(cov[1:], acc[1:], color=colors[i])
            plt.errorbar(cov[1::10], acc[1::10], yerr=acc_std[1::10],
                         label=methods[i], fmt='o', color=colors[i])
        elif method == "c":
            c = [m["c"] for m in res]
            acc = [m["system_loss"] for m in res]
            if std:
                acc_std = [m["std_loss"] for m in res]
            else:
                acc_std = [0 for m in res]
            plt.plot(c[1:], acc[1:], color=colors[i])
            plt.errorbar(c[1::10], acc[1::10], yerr=acc_std[1::10],
                         label=methods[i], fmt='o', color=colors[i])
    plt.legend()
    if method == "cov":
        plt.xlabel("coverage")
    elif method == "c":
        plt.xlabel("deferral cost")
    plt.ylabel("system loss")
    plt.show()

    filename_sp = filename.split("/")
    str_filename = ""
    if len(filename_sp) > 1:
        for i in range(len(filename_sp)-1):
            if not os.path.exists(str_filename+filename_sp[i]):
                logging.info("Creating directory: {}".format(filename_sp[i]))
                os.mkdir(str_filename + filename_sp[i])
            str_filename += filename_sp[i] + "/"

    plt.savefig(filename)
    filename = filename.replace(".pdf", ".tex")
    tikz_save(filename, figure=fig,
              axis_width='\\textwidth', axis_height='\\textwidth')
