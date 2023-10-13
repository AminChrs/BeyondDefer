import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from MyNet.call_net import networks, optimizer_scheduler
from MyMethod.beyond_defer import BeyondDefer
from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from Experiments.basic_parallel import return_res, experiment_parallel
import logging
import warnings
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


def lower_bound_experiment(dataset, dataset_name, epochs, num_classes, device,
                           ):
    # BD
    classifier, human, meta = networks(dataset_name, "BD", device)
    BD = BeyondDefer(10, classifier, human, meta, device)
    optimizer, scheduler = optimizer_scheduler()
    BD.fit(dataset.data_train_loader, dataset.data_val_loader,
           dataset.data_test_loader, num_classes, epochs, optimizer, lr=0.001,
           scheduler=scheduler, verbose=False)
    defer_bound, meta_bound, meta_err = \
        calculate_lower_bounds(meta, classifier, dataset.data_test_loader,
                               num_classes)

    return defer_bound, meta_bound, meta_err


def calculate_lower_bounds(model_meta,
                           model_classifier,
                           dataloader,
                           n_classes):
    model_meta.eval()
    model_classifier.eval()
    h_y_x_m_sum = 0
    err_meta = 0
    total = 0
    y_m_list = np.zeros((1, 2))
    h_y_x_list = np.array([])
    r_optimal = np.array([])
    with torch.no_grad():
        for _, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            hum_preds = hum_preds.to(device)
            one_hot_m = torch.zeros((data_x.size()[0], n_classes))
            one_hot_m[torch.arange(data_x.size()[0]), hum_preds] = 1
            one_hot_m = one_hot_m.to(device)

            outputs_classifier = F.sigmoid(model_classifier(data_x))
            outputs_classifier /= torch.sum(outputs_classifier, dim=1,
                                            keepdim=True)
            outputs_meta = F.sigmoid(model_meta(data_x, one_hot_m))
            outputs_meta /= torch.sum(outputs_meta, dim=1, keepdim=True)

            _, pred_meta = torch.max(outputs_meta.data, 1)

            err_meta += torch.sum(pred_meta != data_y).item()
            # calculate optimal r
            r_optimal_temp = (data_y == hum_preds) * 1
            r_optimal = np.concatenate([r_optimal,
                                        r_optimal_temp.detach().cpu().numpy()])
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
            h_y_x_m_sum += -1 * torch.sum(torch.mul(outputs_meta,
                                                    safe_log2(outputs_meta)
                                                    )).item()
            total += data_x.size()[0]

    # Calculating Meta learner lower bound
    h_y_x_m = h_y_x_m_sum / total

    # Calculating Deferral lower bound
    y_m_list = y_m_list[1:]
    # deferral_lower_bound = \
    #    calculate_defer_lower_bound(y_m_list, h_y_x_list, total, n_classes)
    deferral_lower_bound = \
        calculate_defer_lower_bound_given_r(y_m_list, h_y_x_list,
                                            total, n_classes, r_optimal)

    # Calculating meta error term in the lower bound of meta
    p_err_meta = err_meta / total
    H_b_p_meta = - p_err_meta * np.log2(max(p_err_meta, 1e-10)) - \
        (1 - p_err_meta) * np.log2(max((1 - p_err_meta), 1e-10))
    meta_err = H_b_p_meta + p_err_meta * np.log2(n_classes - 1)

    return (deferral_lower_bound, h_y_x_m, meta_err)


def calculate_h_y_m(y_m_joint_dist):
    p_y_m = y_m_joint_dist / np.sum(y_m_joint_dist)
    p_y_given_m = y_m_joint_dist / np.sum(y_m_joint_dist, axis=0)
    p_y_given_m[p_y_given_m == 0] = 1
    h_y_m = np.sum(-1 * p_y_m * np.log2(p_y_given_m))
    print("H(Y|M) = ", h_y_m)
    return h_y_m


def calculate_defer_lower_bound(y_m_list, h_y_x_list, total, n_classes):
    defer_bound = n_classes
    for i in range(total ** 2):
        # for i in range(10000):
        if i % 1000 == 0:
            logging.error("Iteration: {}".format(i))
        r = np.random.randint(0, 2, total)
        p_defer = np.sum(r)/r.shape[0]
        h_y_x_selected = h_y_x_list[r == 0]
        h_y_x = h_y_x_selected.sum() / h_y_x_selected.shape[0]

        y_m_selected = y_m_list[r == 1]
        y_m_joint_dist = np.zeros((n_classes, n_classes))
        for y_m in y_m_selected:
            y_m_joint_dist[int(y_m[0]), int(y_m[1])] += 1
        h_y_m = calculate_h_y_m(y_m_joint_dist)
        defer_bound_temp = h_y_m * p_defer + h_y_x * (1 - p_defer)
        if defer_bound_temp < defer_bound:
            defer_bound = defer_bound_temp
    # print(defer_bound)
    return defer_bound


def calculate_defer_lower_bound_given_r(y_m_list, h_y_x_list, total, n_classes, r):
    p_defer = np.sum(r)/r.shape[0]
    h_y_x_selected = h_y_x_list[r == 0]
    h_y_x = h_y_x_selected.sum() / h_y_x_selected.shape[0]
    y_m_selected = y_m_list[r == 1]
    y_m_joint_dist = np.zeros((n_classes, n_classes))
    for y_m in y_m_selected:
        y_m_joint_dist[int(y_m[0]), int(y_m[1])] += 1
    h_y_m = calculate_h_y_m(y_m_joint_dist)
    defer_bound = h_y_m * p_defer + h_y_x * (1 - p_defer)
    return defer_bound


def safe_log2(tensor):
    new_tensor = tensor.clone()
    new_tensor[new_tensor < 1e-10] = 1e-10
    return torch.log2(new_tensor)


def plot_lb(results, filename):
    k_range = np.arange(1, len(results) + 1)
    plt.figure()
    defer_bounds = [res_pack[0] for res_pack in results]
    meta_bounds = [res_pack[1] for res_pack in results]
    meta_errs = [res_pack[2] for res_pack in results]
    plt.plot(k_range, defer_bounds, label="Deferral Lower Bound")
    plt.plot(k_range, meta_bounds, label="Meta Learner Lowers Bound")
    plt.plot(k_range, meta_errs, label="Meta Learner Error")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("Error and Lower Bounds")
    plt.show()
    plt.savefig(filename)


def exp_lb_init():
    return return_res()


def exp_lb_loop(res, iter):
    k = iter + 1
    dataset = CifarSynthDataset(k, False, batch_size=512)
    res_pack = lower_bound_experiment(dataset,
                                      "cifar_synth", 1, 10, device)
    return return_res(res_pack=res_pack)


def exp_lb_conc(cls, res):
    filename = "./Results/test/bounds_vs_k.pdf"
    results = []
    for i in range(len(res)):
        results.append(res[i].res_pack)
    plot_lb(results, filename)


def Exp_lb_parallel(iter):

    LB_Exp = experiment_parallel(exp_lb_loop, exp_lb_init, exp_lb_conc,
                                 9,
                                 "./data/lb/")
    LB_Exp.run(parallel=True, iter=iter)
