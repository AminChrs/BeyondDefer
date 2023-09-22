import numpy as np
import sklearn.metrics
import copy
import logging
import sys

sys.path.append("..")

def compute_deferral_metrics_cost_sensitive(data_test, loss_fn):
    """_summary_

    Args:
        data_test (dict): dict data with fields 'defers', 'labels', 'hum_preds', 'preds'

    Returns:
        dict: dict with metrics, 'classifier_all_acc': classifier accuracy on all data
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
    # get system loss
    n_data = data_test["preds"].shape[0]
    data_concat = np.concatenate([np.reshape(data_test["labels"], (n_data, 1)),
                                  np.reshape(data_test["hum_preds"], (n_data, 1)),
                                  np.reshape(data_test["preds"], (n_data, 1)),
                                  np.reshape(data_test["defers"], (n_data, 1))], axis=1)    
    results["system_loss"] = np.sum(np.apply_along_axis(loss_fn, 1, data_concat))
    
    
    return results