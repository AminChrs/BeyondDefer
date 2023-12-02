import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from beyonddefer.MyNet.call_net import networks, optimizer_scheduler
from beyonddefer.MyMethod.CompareConfMeta import CompareConfMeta
from beyonddefer.Datasets.cifar import CifarSynthDatasetEnt
from beyonddefer.Experiments.basic_parallel import return_res, \
    experiment_parallel
import logging
import warnings
import itertools
import os
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


def joint_distribution(y1y2_list, n_classes):
    dist = np.zeros((n_classes, n_classes))
    for i in range(len(y1y2_list)):
        dist[y1y2_list[i][0], y1y2_list[i][1]] += 1
    return dist


class SetFunction():

    def __init__(self, oracle, setmax):
        self.oracle = oracle
        self.setmax = setmax
        self.all_indices = list(range(len(setmax)))

    def min(self, init_indices=[], iter=10):
        indices = init_indices
        if len(init_indices) == 0:
            value_min = self.oracle([])
        else:
            selfsetmaxinit = []
            for i in range(len(init_indices)):
                selfsetmaxinit.append(self.setmax[init_indices[i]])
            value_min = self.oracle(selfsetmaxinit)
        setmax = self.setmax
        for i in range(iter):
            for j in range(len(setmax)):
                if j not in indices:
                    indices_temp = indices + [j]
                    array_in = []
                    for k in range(len(indices_temp)):
                        array_in.append(self.setmax[indices_temp[k]])
                    value_temp = self.oracle(array_in)
                    if value_temp < value_min:
                        value_min = value_temp
                        index_min = j
                        obtained_opt = True
            if not obtained_opt:
                break
            obtained_opt = False
            indices = indices + [index_min]
            # remove index index_min from setmax
            setmax = setmax[:index_min] + setmax[index_min + 1:]
        setmax_indices = []
        for i in range(len(indices)):
            setmax_indices.append(self.setmax[indices[i]])
        return value_min, indices, setmax_indices


def entropy_set_function(r, h_y_x_list, y_m_list, n_classes, verbose=False):
    p_defer = np.sum(r)/len(r)
    idx_r_0 = np.where(r == np.zeros(len(r)))[0]
    idx_r_1 = np.where(r == np.ones(len(r)))[0]
    h_y_x_selected = h_y_x_list[idx_r_0]
    if p_defer == 1.0:
        h_y_x = 0
    else:
        h_y_x = h_y_x_selected.sum() / h_y_x_selected.shape[0]
    assert not np.isnan(h_y_x)
    y_m_selected = y_m_list[idx_r_1]
    y_m_joint_dist = np.zeros((n_classes, n_classes))
    for y_m in y_m_selected:
        y_m_joint_dist[int(y_m[0]), int(y_m[1])] += 1
    if p_defer == 0.0:
        h_y_m = 0
    else:
        h_y_m = calculate_h_y_m(y_m_joint_dist)
    assert not np.isnan(h_y_m)
    defer_bound_temp = h_y_m * p_defer + h_y_x * (1 - p_defer)
    if verbose:
        logging.error("p_defer: " + str(p_defer))
        logging.error("h_y_x: " + str(h_y_x))
        logging.error("h_y_m: " + str(h_y_m))
    return defer_bound_temp


def ent_set_function(set_in, set_max, h_y_x_list, y_m_list, n_classes,
                     verbose=False):
    r = np.zeros(len(set_max))
    for i in range(len(set_in)):
        r[set_in[i]] = 1.0
    entsf = entropy_set_function(r, h_y_x_list, y_m_list, n_classes, verbose)
    return entsf


def lower_bound_experiment(dataset, dataset_name, epochs, num_classes, device,
                           ):
    # BD
    classifier, meta, defer, defer_meta = networks(dataset_name,
                                                   "CompConfMeta", device)
    CCM = CompareConfMeta(10, classifier, meta, defer, defer_meta, device)
    optimizer, scheduler = optimizer_scheduler()
    CCM.fit(dataset.data_train_loader, dataset.data_val_loader,
            dataset.data_test_loader, num_classes, epochs, optimizer, lr=1e-3,
            scheduler=scheduler, verbose=False)
    defer_bound, defer_bound2, meta_bound, meta_err, meta_err2 = \
        calculate_lower_bounds(meta, classifier, dataset.data_test_loader,
                               num_classes)

    return defer_bound, defer_bound2, meta_bound, meta_err, meta_err2


def calculate_lower_bounds(model_meta,
                           model_classifier,
                           dataloader,
                           n_classes):
    model_meta.eval()
    model_classifier.eval()
    h_y_x_m_sum = 0
    h_y_yhat = 0
    err_meta = 0
    total = 0
    y_m_list = np.zeros((1, 2))
    h_y_x_list = np.array([])
    with torch.no_grad():
        for _, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            hum_preds = hum_preds.to(device)
            one_hot_m = torch.zeros((data_x.size()[0], n_classes))
            one_hot_m[torch.arange(data_x.size()[0]), hum_preds] = 1
            one_hot_m = one_hot_m.to(device)

            outputs_classifier = F.softmax(model_classifier(data_x), dim=1)
            outputs_meta = F.softmax(model_meta(data_x, one_hot_m), dim=1)

            _, pred_meta = torch.max(outputs_meta.data, 1)

            err_meta += torch.sum(pred_meta != data_y).item()

            # calculate y_m_joint_dist, H(Y|X=x) list, and H(Y|X=x, M=m)
            y_reshaped = data_y.detach().cpu().numpy().reshape(
                                                        (data_y.size()[0], 1))
            m_reshaped = hum_preds.detach().cpu().numpy().reshape(
                                                    (hum_preds.size()[0], 1))
            y_m_list = np.concatenate([y_m_list,
                                       np.concatenate([y_reshaped,
                                                       m_reshaped],
                                                      axis=1)], axis=0)

            h_y_x_list = np.concatenate([h_y_x_list,
                                         -1 * torch.sum(torch.mul(
                                            outputs_classifier, safe_log2(
                                                outputs_classifier)),
                                                1).detach().cpu().numpy()])
            pred_meta = np.argmax(outputs_meta.detach().cpu().numpy(), axis=1)
            jd = joint_distribution(np.concatenate([y_reshaped,
                                                    pred_meta.reshape(
                                                        (pred_meta.shape[0],
                                                         1))], axis=1),
                                    n_classes)
            if total == 0:
                jd_tot = jd
            else:
                jd_tot += jd

            h_y_x_m_sum += -1 * torch.sum(torch.mul(outputs_meta,
                                                    safe_log2(outputs_meta)
                                                    )).item()
            total += data_x.size()[0]
    h_y_yhat = calculate_h_y_m(jd_tot)
    # Calculating Meta learner lower bound
    h_y_x_m = h_y_x_m_sum / total

    # Calculating Deferral lower bound
    y_m_list = y_m_list[1:]
    dlb, dlb2 = \
        calculate_defer_lower_bound(y_m_list, h_y_x_list, n_classes)

    # Calculating meta error term in the lower bound of meta
    p_err_meta = err_meta / total
    H_b_p_meta = - p_err_meta * np.log2(max(p_err_meta, 1e-10)) - \
        (1 - p_err_meta) * np.log2(max((1 - p_err_meta), 1e-10))
    meta_err = H_b_p_meta + p_err_meta * np.log2(n_classes - 1)

    return (dlb, dlb2, h_y_x_m, meta_err, h_y_yhat)


def calculate_h_y_m(y_m_joint_dist):
    p_y_m = y_m_joint_dist / np.sum(y_m_joint_dist)
    p_y_given_m = y_m_joint_dist / np.sum(y_m_joint_dist, axis=0)
    p_y_given_m[p_y_given_m == 0] = 1
    for i in range(p_y_given_m.shape[0]):
        for j in range(p_y_given_m.shape[1]):
            if np.isnan(p_y_given_m[i, j]):
                p_y_given_m[i, j] = 1
    h_y_m = np.sum(-1 * p_y_m * np.log2(p_y_given_m))
    return h_y_m


def calculate_defer_lower_bound(y_m_list, h_y_x_list, n_classes):
    defer_bound = n_classes
    comb_list = []
    for i in range(n_classes + 1):
        for c in itertools.combinations(range(n_classes), i):
            comb_list.append(c)
    for i in range(2**n_classes):
        # if i % 100 == 0:
        #     logging.error("Iteration: {}".format(i))
        r = [1 if y_m[0] in comb_list[i] else 0 for y_m in y_m_list]
        defer_bound_temp = entropy_set_function(r, h_y_x_list, y_m_list,
                                                n_classes, verbose=True)
        p_defer = np.sum(r) / len(r)
        if defer_bound_temp < defer_bound:
            defer_bound = defer_bound_temp
            logging.error("proportion of defer: {}".format(p_defer))
            logging.error("defer bound: {}".format(defer_bound))
    set_max = np.arange(0, len(r)).tolist()

    def entf(set_in): return ent_set_function(set_in, set_max, h_y_x_list,
                                              y_m_list,
                                              n_classes)
    SetFunction(entf, set_max)
    # defer_bound_2 = MySetFunc.min([], iter=len(set_max))
    defer_bound_2 = 0
    # print(defer_bound)
    return defer_bound, defer_bound_2


def safe_log2(tensor):
    new_tensor = tensor.clone()
    new_tensor[new_tensor < 1e-10] = 1e-10
    return torch.log2(new_tensor)


def plot_lb(results, filename):
    k_range = np.arange(1, len(results) + 1)
    plt.figure()
    defer_bounds = [res_pack[0] for res_pack in results]
    # defer_bounds2 = [res_pack[1] for res_pack in results]
    meta_bounds = [res_pack[2] for res_pack in results]
    meta_errs = [res_pack[3] for res_pack in results]
    meta_bounds2 = [res_pack[4] for res_pack in results]
    plt.plot(k_range, defer_bounds, label="Deferral Lower Bound")
    # plt.plot(k_range, defer_bounds2, label="Deferral Lower Bound Greedy")
    plt.plot(k_range, meta_bounds, label="Meta Learner Lowers Bound")
    plt.plot(k_range, meta_errs, label="Meta Learner Error")
    plt.plot(k_range, meta_bounds2, label="Meta Learner Lower Bound 2")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Error and Lower Bounds")
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


def exp_lb_init():
    return return_res()


def exp_lb_loop(res, iter):
    k = iter + 1
    dataset = CifarSynthDatasetEnt(k, False, batch_size=512)
    res_pack = lower_bound_experiment(dataset,
                                      "cifar_synth", 80, 10, device)
    return return_res(res_pack=res_pack)


def exp_lb_conc(cls, res):
    filename = "./Results/LB/bounds_vs_k.pdf"
    results = []
    for i in range(len(res)):
        results.append(res[i].res_pack)
    plot_lb(results, filename)


def Exp_lb_parallel(iter):

    LB_Exp = experiment_parallel(exp_lb_loop, exp_lb_init, exp_lb_conc,
                                 9,
                                 "./data/lb/")
    LB_Exp.run(parallel=True, iter=iter)
