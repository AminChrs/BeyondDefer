# import logging
import numpy as np
import sklearn.metrics
import copy
import matplotlib.pyplot as plt


def aggregate_plots(xs, accs, x_out, method="max"):

    accs_out = []
    for i, acc in enumerate(accs):
        acc_o = []
        for j in range(len(x_out)):
            if x_out[j] < min(xs[i]) or x_out[j] > max(xs[i]):
                if method == "max" or method == "avg":
                    acc_o.append(-np.inf)
                elif method == "min":
                    acc_o.append(np.inf)
            else:
                acc_o.append(np.interp(x_out[j], xs[i], acc))
        accs_out.append(acc_o)
    accs_out = np.array(accs_out)
    if method == "max":
        accs_out = np.max(accs_out, axis=0)
        return accs_out
    elif method == "min":
        accs_out = np.min(accs_out, axis=0)
        return accs_out
    elif method == "avg":
        accs_out_std = np.std(accs_out, axis=0)
        accs_out = np.mean(accs_out, axis=0)
        return accs_out, accs_out_std



def compute_metalearner_metrics(data_test):
    """_summary_

    Args:
        data_test (dict): dict data with fields 'defers',
        'labels', 'hum_preds', 'preds'

    Returns:
        dict: dict with metrics, 'classifier_all_acc':
        classifier accuracy on all data
    'human_all_acc': human accuracy on all data
    'coverage': how often classifier predicts

    """
    results = {}
    results["classifier_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"], data_test["labels"]
    )
    results["meta_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["meta_preds"], data_test["labels"]
    )
    results["coverage"] = 1 - np.mean(data_test["defers"])
    # get classifier accuracy when defers is 0
    if data_test["defers"].sum() < len(data_test["defers"]):
        results["classifier_nondeferred_acc"] = sklearn.metrics.accuracy_score(
            data_test["preds"][data_test["defers"] == 0],
            data_test["labels"][data_test["defers"] == 0],
        )
    else:
        results["classifier_nondeferred_acc"] = 0
    # get human accuracy when defers is 1
    results["meta_deferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["meta_preds"][data_test["defers"] == 1],
        data_test["labels"][data_test["defers"] == 1],
    )
    # get system accuracy
    results["system_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"] * (1 - data_test["defers"])
        + data_test["meta_preds"] * (data_test["defers"]),
        data_test["labels"],
    )
    return results


def cov_vs_acc_AFE(data_test, method="cov"):
    rej_scores = np.unique(data_test["rej_score"])
    if method == "c":
        rej_scores_quantiles = np.arange(0, 1, 0.01)
    else:
        rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
    all_metrics = []
    all_metrics.append({"coverage": 0, "system_acc": 0})
    for q in rej_scores_quantiles:
        metrics = {}
        defers = (data_test["rej_score"] > q).astype(int)
        metrics["coverage"] = 1 - np.mean(defers)
        loss = data_test["loss_meta"]*defers +\
            data_test["loss_class"]*(1-defers)
        accuracy = 1 - loss
        metrics["system_acc"] = np.mean(accuracy)
        all_metrics.append(metrics)
    return all_metrics

def cov_vs_acc_meta(data_test, method="cov"):
    rej_scores = np.unique(data_test["rej_score"])
    if method == "c":
        rej_scores_quantiles = np.arange(0, 1, 0.01)
    else:
        rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
    all_metrics = []
    all_metrics.append(compute_metalearner_metrics(data_test))
    for q in rej_scores_quantiles:
        defers = (data_test["rej_score"] > q).astype(int)
        copy_data = copy.deepcopy(data_test)
        copy_data["defers"] = defers
        metrics = compute_metalearner_metrics(copy_data)
        all_metrics.append(metrics)
    return all_metrics


def cov_vs_acc_add(data_test, method="cov", loss_matrix=None):
    rej_scores1 = np.unique(data_test["rej_score1"])
    rej_scores2 = np.unique(data_test["rej_score2"])
    if method == "c":
        rej_scores1_quantiles = np.arange(0, 1, 0.01)
        rej_scores2_quantiles = np.arange(0, 1, 0.01)
    else:
        rej_scores1_quantiles = \
            np.quantile(rej_scores1, np.linspace(0, 1, 100))
        rej_scores2_quantiles = \
            np.quantile(rej_scores2, np.linspace(0, 1, 100))
    all_metrics = []
    all_metrics.append(compute_additional_defer_metrics(data_test,
                       loss_matrix=loss_matrix))
    accs = []
    covs = []
    losses = []
    for q in rej_scores1_quantiles:
        # logging.info("q: {}".format(q))
        tot_metrics = []
        for q2 in rej_scores2_quantiles:
            if method == "c":
                q2 = q
            stack = np.vstack([np.zeros(len(data_test["rej_score1"])),
                               data_test["rej_score1"] - q,
                               data_test["rej_score2"] - q2])
            defers = np.argmax(stack, axis=0)
            copy_data = copy.deepcopy(data_test)
            copy_data["defers"] = defers
            metrics = compute_additional_defer_metrics(copy_data,
                                                       loss_matrix=loss_matrix)
            tot_metrics.append(metrics)
            if method == "c":
                break
        if method == "c":
            accs.append(tot_metrics[0]["system_acc"])
            covs.append(tot_metrics[0]["coverage"])
        elif method == "cov":
            accs.append([m["system_acc"] for m in tot_metrics])
            covs.append([m["coverage"] for m in tot_metrics])
            if loss_matrix is not None:
                losses.append([m["system_loss"] for m in tot_metrics])
    if method == "c":
        accs_out = accs
        x_out = covs
    else:
        x_out = np.arange(0, 1, 0.01)
        accs_out = aggregate_plots(covs, accs, x_out, method="max")
        if loss_matrix is not None:
            losses_out = aggregate_plots(covs, losses, x_out, method="min")
    for i in range(len(accs_out)):
        if loss_matrix is not None:
            all_metrics.append({"system_acc": accs_out[i], "coverage":
                                x_out[i], "system_loss": losses_out[i]})
        else:
            all_metrics.append({"system_acc": accs_out[i], "coverage":
                                x_out[i]})
    return all_metrics


def plot_cov_vs_acc(data_test):

    all_metrics = cov_vs_acc_meta(data_test)
    cov = [m["coverage"] for m in all_metrics]
    acc = [m["system_acc"] for m in all_metrics]
    plt.plot(cov, acc)
    plt.xlabel("coverage")
    plt.ylabel("system accuracy")
    plt.title("coverage vs system accuracy")
    plt.show()
    plt.savefig("Results/coverage_vs_system_acc_BD.pdf")


def compute_coverage_v_acc_curve(data_test, method="cov", loss_matrix=None):
    """

    Args:
        data_test (dict): dict data with field   {'defers': defers_all,
        'labels': truths_all, 'hum_preds': hum_preds_all,
        'preds': predictions_all, 'rej_score': rej_score_all,
        'class_probs': class_probs_all}

    Returns:
        data (list): compute_deferral_metrics(data_test_modified) on
        different coverage levels, first element of list
        is compute_deferral_metrics(data_test)
    """
    # get unique rejection scores
    rej_scores = np.unique(data_test["rej_score"])
    # sort by rejection score
    # get the 100 quantiles for rejection scores
    if method == "c":
        rej_scores_quantiles = np.arange(0, 1, 0.01)
    else:
        rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
    # for each quantile, get the coverage and accuracy by getting a
    # new deferral decision
    all_metrics = []
    all_metrics.append(compute_deferral_metrics(data_test,
                                                loss_matrix=loss_matrix))
    for q in rej_scores_quantiles:
        # get deferral decision
        defers = (data_test["rej_score"] > q).astype(int)
        copy_data = copy.deepcopy(data_test)
        copy_data["defers"] = defers
        # compute metrics
        metrics = compute_deferral_metrics(copy_data, loss_matrix=loss_matrix)
        all_metrics.append(metrics)
    return all_metrics


def compute_additional_defer_metrics(data_test, loss_matrix=None):
    """_summary_

    Args:
        data_test (dict): dict data with fields 'defers', 'labels',
        'hum_preds', 'preds'

    Returns:
        dict: dict with metrics, 'classifier_all_acc': classifier accuracy on
        all data
    'human_all_acc': human accuracy on all data
    'coverage': how often classifier predicts
    

    """
    results = {}
    results["classifier_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"], data_test["labels"]
    )
    results["meta_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["meta_preds"], data_test["labels"]
    )
    results["human_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["human_preds"], data_test["labels"]
    )
    results["coverage"] = 1 - (np.sum(data_test["defers"] == 1)
                               + np.sum(data_test["defers"] == 2)) / len(
                                data_test["defers"])
    results["defer_prop"] = np.sum(data_test["defers"] == 1) / len(
                                    data_test["defers"])
    results["defer_prop_meta"] = np.sum(data_test["defers"] == 2) / len(
                                    data_test["defers"])
    # get classifier accuracy when defers is 0
    if data_test["defers"].sum() < len(data_test["defers"]):
        results["classifier_nondeferred_acc"] = sklearn.metrics.accuracy_score(
            data_test["preds"][data_test["defers"] == 0],
            data_test["labels"][data_test["defers"] == 0],
        )
    else:
        results["classifier_nondeferred_acc"] = 0
    # get human accuracy when defers is 1
    results["meta_deferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["meta_preds"][data_test["defers"] == 2],
        data_test["labels"][data_test["defers"] == 2],
    )
    results["human_deferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["human_preds"][data_test["defers"] == 1],
        data_test["labels"][data_test["defers"] == 1],
    )
    # get system accuracy
    results["system_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"] * (data_test["defers"] == 0)
        + data_test["human_preds"] * (data_test["defers"] == 1)
        + data_test["meta_preds"] * (data_test["defers"] == 2),
        data_test["labels"],
    )
    if loss_matrix is not None:
        loss_matrix = loss_matrix.cpu().numpy()
        num_classes = loss_matrix.shape[0]
        labels_oh = np.eye(num_classes)[data_test["labels"]]
        preds = data_test["preds"] * (data_test["defers"] == 0) \
            + data_test["human_preds"] * (data_test["defers"] == 1) \
            + data_test["meta_preds"] * (data_test["defers"] == 2)
        preds_oh = np.eye(num_classes)[preds]
        cost_y = np.matmul(labels_oh, loss_matrix)
        loss = cost_y * preds_oh
        loss = np.sum(loss, axis=1)
        results["system_loss"] = np.mean(loss)
    return results


def compute_deferral_metrics(data_test, loss_matrix=None):
    """_summary_

    Args:
        data_test (dict): dict data with fields 'defers', 'labels',
        'hum_preds',
        'preds'

    Returns:
        dict: dict with metrics, 'classifier_all_acc': classifier accuracy on
        all data
    'human_all_acc': human accuracy on all data
    'coverage': how often classifier predicts

    """
    results = {}
    results["classifier_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"], data_test["labels"]
    )
    results["human_all_acc"] = sklearn.metrics.accuracy_score(
        data_test["hum_preds"], data_test["labels"]
    )
    results["coverage"] = 1 - np.mean(data_test["defers"])
    # get classifier accuracy when defers is 0
    results["classifier_nondeferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"][data_test["defers"] == 0],
        data_test["labels"][data_test["defers"] == 0],
    )
    # get human accuracy when defers is 1
    results["human_deferred_acc"] = sklearn.metrics.accuracy_score(
        data_test["hum_preds"][data_test["defers"] == 1],
        data_test["labels"][data_test["defers"] == 1],
    )
    # get system accuracy
    results["system_acc"] = sklearn.metrics.accuracy_score(
        data_test["preds"] * (1 - data_test["defers"])
        + data_test["hum_preds"] * (data_test["defers"]),
        data_test["labels"],
    )
    if loss_matrix is not None:
        loss_matrix = loss_matrix.cpu().numpy()
        num_classes = loss_matrix.shape[0]
        labels_oh = np.eye(num_classes)[data_test["labels"]]
        preds = data_test["preds"] * (1 - data_test["defers"])\
            + data_test["hum_preds"] * (data_test["defers"])
        preds_oh = np.eye(num_classes)[preds]
        cost_y = np.matmul(labels_oh, loss_matrix)
        loss = cost_y * preds_oh
        loss = np.sum(loss, axis=1)
        results["system_loss"] = np.mean(loss)
    return results
