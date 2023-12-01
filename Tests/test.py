from metrics.metrics import cov_vs_acc_meta, cov_vs_acc_add, cov_vs_acc_AFE
from Feature_Acquisition.active import IndexedDataset, ActiveDataset, AFE
from MyNet.call_net import networks, optimizer_scheduler
from human_ai_defer.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_defer.datasetsdefer.hatespeech import HateSpeech
from human_ai_defer.datasetsdefer.cifar_h import Cifar10h
from human_ai_defer.datasetsdefer.imagenet_16h import ImageNet16h
from human_ai_defer.methods.realizable_surrogate import RealizableSurrogate
from human_ai_defer.helpers.metrics import compute_coverage_v_acc_curve
from Experiments.basic import plot_cov_vs_acc, plot_cov_vs_cost
from MyMethod.additional_cost import AdditionalCost
from BL.lce_cost import LceCost
from BL.compare_confidence_cost import CompareConfCost
from BL.one_v_all_cost import OVACost
from MyMethod.CompareConfMeta import CompareConfMeta
from MyMethod.CompareConfMetaCost import CompareConfMetaCost
from MyNet.networks import MetaNet
from MyMethod.beyond_defer import BeyondDefer
from MyMethod.additional_defer import AdditionalBeyond
from human_ai_defer.networks.cnn import NetSimple
from MyMethod.learned_beyond import LearnedBeyond
from MyMethod.learned_additional import LearnedAdditional
from Datasets.cifar import CifarSynthDatasetEnt
# from Experiments.comparison import Exp_lb_parallel
from Experiments.comparison import SetFunction
# from metrics.metrics import plot_cov_vs_acc
from Experiments.basic_parallel import experiment_parallel, return_res
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
import logging
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

logging.getLogger().setLevel(logging.INFO)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# np.seterr(invalid='raise')


def test_indexed():

    Test_Text = False

    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_indexed = IndexedDataset(Dataset_CIFAR)

    for i in range(len(Dataset_CIFAR_indexed)):
        index, (x, y, m) = Dataset_CIFAR_indexed[i]
        assert x.shape == (3, 32, 32)
        assert y >= 0 and y < 10
        assert m >= 0 and m < 10
        assert index == i

    if Test_Text:
        data_dir = './human_ai_defer/data/'
        Dataset_Hate = HateSpeech(data_dir, True, False, 'random_annotator',
                                  device)
        Dataset_Hate_indexed = IndexedDataset(Dataset_Hate)
        for i in range(len(Dataset_Hate_indexed)):
            index, (x, y, m) = Dataset_Hate_indexed[i]
            assert x.shape == torch.Size([384])
            assert y.shape == torch.Size([])
            assert m.shape == torch.Size([])
            assert y.item() >= 0 and y.item() < 10
            assert m.item() >= 0 and m.item() < 10
            assert index == i
    logging.info("Test Indexed Dataset Passed!")


def test_active_mask():

    Test_Text = False

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)
    Dataset_CIFAR_Active.mask(125)
    assert Dataset_CIFAR_Active.mask_label(125) == 1
    assert Dataset_CIFAR_Active.mask_label(126) == 0

    # Text
    if Test_Text:
        data_dir = './human_ai_defer/data/'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Dataset_Hate = HateSpeech(data_dir, True, False, 'random_annotator',
                                  device)
        Dataset_Hate_Active = ActiveDataset(Dataset_Hate)
        Dataset_Hate_Active.mask(125)
        assert Dataset_Hate_Active.mask_label(125) == 1
        assert Dataset_Hate_Active.mask_label(126) == 0
    logging.info("Test Active Dataset Passed!")


def test_active_query():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)
    global idx_highest
    idx_highest = 0

    def criterion(loader_unlabeled, _):
        indices = []
        Loss = []
        loss_start = 0
        global idx_highest
        for batch, (idx, (x, y, m)) in enumerate(loader_unlabeled):
            for i in idx:
                indices.append(i)
                Loss.append(loss_start)
                loss_start += 0.1
                idx_highest = i
        # Convert to tensor
        Loss = torch.tensor(Loss)
        return Loss, indices
    Dataset_CIFAR_Active.Query(criterion, pool_size=0, query_size=1)
    Dataset_CIFAR_Active.Query(criterion, pool_size=0, query_size=10)
    assert idx_highest != 0
    assert Dataset_CIFAR_Active.mask_label(idx_highest) == 1
    assert Dataset_CIFAR_Active.mask_label(idx_highest - 1) == 0
    logging.info("Test Active Query Passed!")


def test_AFE_loss():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)

    # Classifier
    Classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)

    # AFE
    AFE_CIFAR = AFE(Classifier, Meta, device)
    index = 0
    for i in range(len(Dataset_CIFAR_Active.train_dataset)):
        idx, (x, y, m) = Dataset_CIFAR_Active.train_dataset[i]
        x = torch.tensor(x).unsqueeze(0).to(device)
        y = torch.tensor(y).unsqueeze(0).to(device)
        m = torch.tensor(m).unsqueeze(0).to(device)
        # turn m into one hot
        m = torch.nn.functional.one_hot(m, 10).float()
        assert x.shape == torch.Size([1, 3, 32, 32])
        assert y.shape == torch.Size([1])
        assert m.shape == torch.Size([1, 10])

        #
        assert AFE_CIFAR.AFELoss(Classifier(x), Meta(x, m)).shape == \
            torch.Size([1])
        index += 1
        break
    logging.info("Test AFE Passed!")


def test_AFE_loss_loaders():
    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)

    len_1 = 100
    len_2 = 50

    dataset_actual = Dataset_CIFAR_Active.train_dataset
    dataset_Indexed = IndexedDataset(Dataset_CIFAR)
    indices = np.random.choice(len(dataset_actual), 100, replace=False)
    inv_indices = np.setdiff1d(np.arange(len(dataset_actual)), indices)
    inv_indices = inv_indices[:len_2]
    indices = indices[:len_1]
    dataset_loader1 = torch.utils.data.DataLoader(
        dataset_actual, batch_size=1,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
    dataset_loader2 = torch.utils.data.DataLoader(
        dataset_Indexed, batch_size=1,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(inv_indices))

    # Classifier
    Classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)

    # AFE
    AFE_CIFAR = AFE(Classifier, Meta, device)
    KL_loss, indices = AFE_CIFAR.AFELoss_loaders(dataset_loader1,
                                                 dataset_loader2, 10)
    assert KL_loss.shape == torch.Size([len_1])
    assert len(indices) == len_1
    logging.info("Test AFE Loss Loaders Passed!")


def test_Meta_model():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    train_loader = Dataset_CIFAR.data_train_loader

    # Classifier
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)

    for batch, (x, y, m) in enumerate(train_loader):
        x = x.to(device)
        m = m.to(device)
        assert x.shape == torch.Size([512, 3, 32, 32])
        assert y.shape == torch.Size([512])
        m = torch.nn.functional.one_hot(m, 10).float()
        assert m.shape == torch.Size([512, 10])

        assert Meta(x, m).shape == torch.Size([512, 10])
        break
    logging.info("Test Meta Model Passed!")


def test_AFE_fit_epochs():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)
    Dataset_CIFAR_indexed = IndexedDataset(Dataset_CIFAR)
    dataloader_meta = DataLoader(Dataset_CIFAR_indexed, batch_size=10)

    # Classifier and Meta
    Classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)

    # AFE
    AFE_CIFAR = AFE(Classifier, Meta, device)
    optim = torch.optim.Adam(Classifier.parameters(), lr=0.001)
    optim_meta = torch.optim.Adam(Meta.parameters(), lr=0.001)

    AFE_CIFAR.fit_Eo_epoch(dataloader_meta, 10,
                           optim_meta, verbose=True)
    AFE_CIFAR.fit_El_epoch(Dataset_CIFAR_Active.data_train_loader, 10, optim,
                           verbose=True)

    logging.info("Test AFE Fit Classifier Passed!")


def test_AFE_CE_loss():

    # Image Dataset
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)

    # Classifier
    Classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)

    # AFE
    AFE_CIFAR = AFE(Classifier, Meta, device)

    # read one batch
    _, (x, y, _) = next(enumerate(Dataset_CIFAR_Active.data_train_loader))
    x = x.to(device)
    y = y.to(device)
    yhat = Classifier(x)
    assert yhat.shape == torch.Size([512, 10])
    assert y.shape == torch.Size([512])
    assert AFE_CIFAR.Loss(yhat, y).shape == torch.Size([512])
    logging.info("Test AFE CE Loss Passed!")


def test_AFE_fit_Eu():

    # Image Dataset
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)

    # Classifier
    Classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)
    Meta.weight_init()

    # AFE
    AFE_CIFAR = AFE(Classifier, Meta, device)

    # optimizer_meta = torch.optim.SGD(Meta.parameters(), 0.001, #0.001
    #                             momentum=0.9, nesterov=True,
    #                             weight_decay=5e-4)

    optimizer_meta = torch.optim.Adam(Meta.parameters(), lr=0.001,
                                      weight_decay=5e-4)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_meta,
                                                           34000 * 150)

    # optimizer_meta = torch.optim.Adam(Meta.parameters(), lr=0.01)

    def criterion(loader_unlabeled, _):
        indices = []
        Loss = []
        loss_start = 0
        global idx_highest
        for batch, (idx, (x, y, m)) in enumerate(loader_unlabeled):
            for i in idx:
                indices.append(i)
                Loss.append(loss_start)
                loss_start += 0.1
                idx_highest = i
        # Convert to tensor
        Loss = torch.tensor(Loss)
        return Loss, indices
    Dataset_CIFAR_Active.Query(criterion, pool_size=0, query_size=34000)
    Dataset_CIFAR_Active.Query(criterion, pool_size=0, query_size=1)
    AFE_CIFAR.fit_Eu(1, Dataset_CIFAR_Active,
                     10, optimizer_meta, verbose=True,
                     scheduler_meta=scheduler)  # 150
    logging.info("Test AFE Fit Eu Passed!")


def test_AFE_fit():

    # Image Dataset
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)
    # find data loader length
    length = len(Dataset_CIFAR_Active.data_train_loader.dataset)

    # Classifier
    Classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)

    # AFE
    AFE_CIFAR = AFE(Classifier, Meta, device)

    def scheduler(z, l):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, l)

    AFE_CIFAR.fit(Dataset_CIFAR_Active,
                  10, 1, lr=0.001, verbose=True,
                  query_size=int(np.floor(length)),
                  num_queries=1, scheduler_classifier=scheduler,
                  scheduler_meta=scheduler)
    with open('AFE_CIFAR_report.json', 'w') as fp:
        json.dump(AFE_CIFAR.report, fp)
    with open('AFE_CIFAR_report.json', 'r') as fp:
        AFE_CIFAR.report = json.load(fp)

    logging.info("Test AFE fit passed!")


def test_Query_unnumbered():
    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)
    global idx_highest
    idx_highest = 0

    def criterion(loader_unlabeled, _):
        indices = []
        Loss = []
        loss_start = 0
        global idx_highest
        for batch, (idx, (x, y, m)) in enumerate(loader_unlabeled):
            for i in idx:
                indices.append(i)
                Loss.append(loss_start)
                loss_start += 0.1
                idx_highest = i
        # Convert to tensor
        Loss = torch.tensor(Loss)
        return Loss, indices

    # Classifier
    Classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)

    # AFE
    AFE_CIFAR = AFE(Classifier, Meta, device)

    min_idx, len_val = Dataset_CIFAR_Active.Query_unnumbered(
                                        criterion, AFE_CIFAR.loss_defer)
    assert min_idx.shape == torch.Size([])
    assert min_idx < len_val
    logging.info("Test Query Unnumbered passed!")


def test_Query_test():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)
    global idx_highest
    idx_highest = 0

    def criterion(loader_unlabeled, _):
        indices = []
        Loss = []
        loss_start = 0
        global idx_highest
        for batch, (idx, (x, y, m)) in enumerate(loader_unlabeled):
            for i in idx:
                indices.append(i)
                Loss.append(loss_start)
                loss_start += 0.1
                idx_highest = i
        # Convert to tensor
        Loss = torch.tensor(Loss)
        return Loss, indices

    # Classifier
    Classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)

    # AFE
    AFE_CIFAR = AFE(Classifier, Meta, device)

    loss = Dataset_CIFAR_Active.Query_test(criterion,
                                           AFE_CIFAR.loss_defer, 10)
    loss = loss[0]
    logging.info(loss)
    assert isinstance(loss, float)
    assert loss >= 0
    assert not np.isnan(loss)
    logging.info("Test Query Test passed!")


def test_iteration_report():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    Dataset_CIFAR_Active = ActiveDataset(Dataset_CIFAR)

    # Classifier
    Classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)

    # AFE
    AFE_CIFAR = AFE(Classifier, Meta, device)

    AFE_CIFAR.report_iteration(0, Dataset_CIFAR_Active, 10)
    assert len(AFE_CIFAR.report) == 1
    assert AFE_CIFAR.report[0]["query_num"] == 0
    assert AFE_CIFAR.report[0]["defer_size"] >= 0
    assert AFE_CIFAR.report[0]["loss_defer"] > 0.0
    if not os.path.isfile('AFE_CIFAR_report.json'):
        with open('AFE_CIFAR_report.json', 'w') as fp:
            json.dump(AFE_CIFAR.report, fp)
    with open('AFE_CIFAR_report.json', 'r') as fp:
        AFE_CIFAR_report = json.load(fp)
        meta_loss = np.zeros(len(AFE_CIFAR_report))
        class_loss = np.zeros(len(AFE_CIFAR_report))
        defer_loss = np.zeros(len(AFE_CIFAR_report))
        for i in range(len(AFE_CIFAR_report)):
            meta_loss[i] = \
                AFE_CIFAR_report[i]["test_metrics_meta"]["meta_all_acc"]
            class_loss[i] = \
                AFE_CIFAR_report[i]["test_metrics_class"]["classifier_all_acc"]
            defer_loss[i] = AFE_CIFAR_report[i]["loss_defer"]

    # Plot
    plt.figure()
    plt.plot(meta_loss, label="meta_loss")
    plt.plot(class_loss, label="class_loss")
    plt.plot(defer_loss, label="defer_loss")
    plt.legend()
    plt.savefig("AFE_CIFAR_report.pdf", format="pdf")

    logging.info("Test Iteration Report passed!")


def test_OVA_loss():
    # y are n number between 0 and 9
    pred = torch.randn(10, 10)

    # Initialize method
    model_classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    model_human = NetSimple(10, 50, 50, 100, 20).to(device)
    model_meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                         remove_layers=["fc3", "softmax"]).to(device)
    BD = BeyondDefer(10, model_classifier, model_human, model_meta, device)

    loss = BD.LossOVA(pred, 1)
    loss2 = BD.LossOVA(pred, -1)
    assert isinstance(loss, torch.Tensor)
    assert not np.isnan(loss.cpu().numpy()).any()
    assert isinstance(loss2, torch.Tensor)
    assert not np.isnan(loss2.cpu().numpy()).any()
    assert loss.shape == torch.Size([10, 10])
    for i in range(10):
        for j in range(10):
            assert loss[i, j] >= 0
            assert loss2[i, j] >= 0

    logging.info("Test OVA Loss passed!")


def test_BD_loss():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)

#     Initialize method
    model_classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    model_human = NetSimple(10, 50, 50, 100, 20).to(device)
    model_meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                         remove_layers=["fc3", "softmax"]).to(device)
    BD = BeyondDefer(10, model_classifier, model_human, model_meta, device)

    x, y, m = next(iter(Dataset_CIFAR.data_train_loader))
    x = x.to(device)
    y = y.to(device)
    m = m.to(device)
    # make m one-hot
    m_oh = torch.nn.functional.one_hot(m, num_classes=10).float()
    assert m_oh.shape == torch.Size([512, 10])

    model_pred = model_classifier(x)
    human_pred = model_human(x)
    meta_pred = model_meta(x, m_oh)

    loss = BD.surrogate_loss(model_pred, human_pred, meta_pred, m, y)
    print("loss: ", loss)
    assert isinstance(loss, torch.Tensor)
    assert not np.isnan(loss.detach().cpu().numpy()).any()
    assert loss.shape == torch.Size([])
    assert loss >= 0
    print("Test BD Loss passed!")


def test_BD_fit_epoch():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)

    # Initialize method
    model_classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    model_human = NetSimple(10, 50, 50, 100, 20).to(device)
    model_meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                         remove_layers=["fc3", "softmax"]).to(device)

    BD = BeyondDefer(10, model_classifier, model_human, model_meta, device)

    # Fit
    params = list(model_classifier.parameters()) +\
        list(model_human.parameters()) + \
        list(model_meta.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
    BD.fit_epoch(Dataset_CIFAR.data_train_loader, 10, optimizer,
                 verbose=True)
    print("Test BD fit epoch passed!")


def test_BD_fit():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)

    # Initialize method
    model_classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    model_human = NetSimple(10, 50, 50, 100, 20).to(device)
    model_meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                         remove_layers=["fc3", "softmax"]).to(device)

    BD = BeyondDefer(10, model_classifier, model_human, model_meta, device)

    # Fit
    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       weight_decay=0.0005)
    BD.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
           Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)  # 80
    print("Test BD fit passed!")


def test_BD_fit_CIFAR10h():

    # image
    dataset = Cifar10h(False, data_dir='./data')

    # models
    classifier, human, meta = networks("cifar_10h", "BD", device, sp_epochs=1)

    # BD
    BD = BeyondDefer(10, classifier, human, meta, device)
    optimizer, scheduler = optimizer_scheduler()
    # fit
    BD.fit(dataset.data_train_loader, dataset.data_val_loader,
           dataset.data_test_loader, 10, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)  # 80
    plot_cov_vs_acc(BD.test(dataset.data_test_loader, 10))
    print("Test BD on CIFAR-10H fit passed!")


def test_BD_fit_Imagenet():

    # image
    dataset = ImageNet16h(False, data_dir="./data/osfstorage-archive/",
                          noise_version="110", batch_size=32, test_split=0.2,
                          val_split=0.01)

    # models
    classifier, human, meta = networks("imagenet", "BD", device)

    # BD
    BD = BeyondDefer(10, classifier, human, meta, device)
    _, scheduler = optimizer_scheduler()

    def optimizer(params, lr):
        return torch.optim.AdamW(params, lr=lr)
    # fit
    BD.fit(dataset.data_train_loader, dataset.data_val_loader,
           dataset.data_test_loader, 16, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)  # 80
    plot_cov_vs_acc(BD.test(dataset.data_test_loader, 16))
    print("Test BD on Imagenet fit passed!")


def test_RS_Imagenet():

    # image
    dataset = ImageNet16h(False, data_dir="./data/osfstorage-archive/",
                          noise_version="110", batch_size=32, test_split=0.2,
                          val_split=0.01)

    # models
    model = networks("imagenet", "RS", device)

    # RS
    Reallizable_Surr = RealizableSurrogate(1, 300, model, device, True)
    optimizer, scheduler = optimizer_scheduler()
    Reallizable_Surr.fit_hyperparam(
        dataset.data_train_loader,
        dataset.data_val_loader,
        dataset.data_test_loader,
        epochs=1,
        optimizer=optimizer,
        scheduler=scheduler,
        lr=0.001,
        verbose=False,
        test_interval=1,
    )  # 100

    print("Test RS on Imagenet fit passed!")


def test_BD_Hatespeech():

    # image
    dataset = HateSpeech("./data/", True, False, 'random_annotator', device)

    # models
    classifier, human, meta = networks("hatespeech", "BD", device)

    # BD
    BD = BeyondDefer(10, classifier, human, meta, device)
    optimizer, scheduler = optimizer_scheduler()
    # fit
    BD.fit(dataset.data_train_loader, dataset.data_val_loader,
           dataset.data_test_loader, 3, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)  # 200
    plot_cov_vs_acc(BD.test(dataset.data_test_loader, 4))
    print("Test BD on HateSpeech fit passed!")


def test_RS_Hatespeech():

    # image
    dataset = HateSpeech("./data/", True, False, 'random_annotator', device)

    # models
    model = networks("hatespeech", "RS", device)

    # RS
    Reallizable_Surr = RealizableSurrogate(1, 2, model, device, True)
    optimizer, scheduler = optimizer_scheduler()
    Reallizable_Surr.fit_hyperparam(
        dataset.data_train_loader,
        dataset.data_val_loader,
        dataset.data_test_loader,
        epochs=1,
        optimizer=optimizer,
        scheduler=scheduler,
        lr=0.001,
        verbose=False,
        test_interval=1,
    )  # 100
    cov_vs_acc = compute_coverage_v_acc_curve(
        Reallizable_Surr.test(dataset.data_test_loader))

    cov = [m["coverage"] for m in cov_vs_acc]
    acc = [m["system_acc"] for m in cov_vs_acc]
    plt.plot(cov, acc)
    plt.xlabel("coverage")
    plt.ylabel("system accuracy")
    plt.title("coverage vs system accuracy")
    plt.show()
    plt.savefig("coverage_vs_system_acc_RS.pdf")

    print("Test RS on HateSpeech fit passed!")


def test_parallel():

    # init
    def init():
        # logging.info("init")
        arr = np.arange(1000)
        return arr

    # for loop
    def for_loop(arr, iter):
        x2x = (arr[iter], arr[iter]**2)
        return return_res(x2x=x2x)

    # last
    def last(cls, res):
        x2 = 0.0
        x = 0.0
        for res_i in res:
            x2 += res_i.x2x[1]
            x += res_i.x2x[0]
        return (x2)/len(res)-(x/len(res))**2

    # parallel
    Parallel = experiment_parallel(for_loop, init, last, 1000,
                                   "data/parallel_test/")
    for i in range(1001):
        Parallel.run(parallel=True, iter=i)
    pres = Parallel.result
    print("Serializing...")
    Parallel.run(parallel=False)
    sres = Parallel.result

    # assert
    logging.info("pres: {}".format(pres))
    assert pres is not None
    assert pres == sres
    assert pres > 0
    logging.info("Test parallel passed!")


def test_return_res():

    a = "Hi"
    b = 2.5

    def xx():
        return 2.2
    c = xx

    res = return_res(a=a, b=b, c=c)
    assert res.a == a
    assert res.b == b
    assert res.c() == c()
    logging.info("Test return_res passed!")


def test_compute_meta():

    rej_score = np.random.rand(1000, 1)*2-1
    zeros = np.zeros((1000, 1))
    # generate 1 with prob rej_score*0.5+0.5
    meta_preds = np.random.rand(1000, 1)
    meta_preds = (meta_preds > rej_score*0.5+0.5).astype(int)
    preds = 1-meta_preds
    data = {
            "defers": zeros,
            "labels": zeros,
            "meta_preds": meta_preds,
            "preds": preds,
            "rej_score": rej_score,
            "class_probs": zeros,
        }
    # res = compute_metalearner_metrics(data)
    res = cov_vs_acc_meta(data)
    for i in range(len(res)):
        logging.info("coverage: {}, system_acc: {}".format(res[i]["coverage"],
                     res[i]["system_acc"]))


def test_additional_defer_loss():

    # image
    dataset = CifarSynthDataset(5, False, batch_size=512)

    # models
    classifier, human, meta = networks("cifar_synth", "Additional", device)

    # Additional
    Add = AdditionalBeyond(10, classifier, human, meta, device)

    x, y, m = next(iter(dataset.data_train_loader))
    # make m one-hot
    m_oh = torch.nn.functional.one_hot(m, num_classes=10).float()
    x = x.to(device)
    y = y.to(device)
    m = m.to(device)
    m_oh = m_oh.to(device)
    assert m_oh.shape == torch.Size([512, 10])
    model_pred = classifier(x)
    human_pred = human(x)
    meta_pred = meta(x, m_oh)

    loss = Add.surrogate_loss(model_pred, human_pred, meta_pred, m, y)
    print("loss: ", loss)
    assert isinstance(loss, torch.Tensor)
    assert not np.isnan(loss.detach().cpu().numpy()).any()
    assert loss.shape == torch.Size([])
    assert loss >= 0
    print("Test Additional Loss passed!")


def test_additional_defer_fit():

    # Image
    expert_k = 9
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)

    # Initialize method
    classifier, human, meta = networks("cifar_synth", "Additional", device)

    AB = AdditionalBeyond(10, classifier, human, meta, device)

    # Fit
    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    AB.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
           Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)  # 80
    test_data = AB.test(Dataset_CIFAR.data_test_loader, 10)
    plot_cov_vs_acc(test_data)
    print("Test Additional fit passed!")


def test_BCE_loss_vs_OvA():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    # Initialize method
    classifier, human, meta = networks("cifar_synth", "Additional", device)

    AB = AdditionalBeyond(10, classifier, human, meta, device)

    for i, (x, y, m) in enumerate(Dataset_CIFAR.data_train_loader):
        logging.info("x shape: {}".format(x.shape))
        logging.info("y shape: {}".format(y.shape))
        logging.info("m shape: {}".format(m.shape))
        # one hot m
        m_oh = torch.nn.functional.one_hot(m, num_classes=10).float()

        l1 = AB.surrogate_loss(classifier(x.to(device)),
                               human(x.to(device)),
                               meta(x.to(device), m_oh.to(device)),
                               m.to(device), y.to(device))
        l2 = AB.surrogate_loss_bce(classifier(x.to(device)),
                                   human(x.to(device)),
                                   meta(x.to(device), m_oh.to(device)),
                                   m.to(device), y.to(device))
        assert l1.shape == l2.shape
        logging.info("Norm of difference: {}".format((l1-l2).norm()))
        logging.info("l1: {}, l2: {}".format(l1, l2))
        assert (l1 - l2).norm() < 1e-5


def test_cov_vs_acc():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    # Initialize method
    classifier, human, meta = networks("cifar_synth", "Additional", device)

    AB = AdditionalBeyond(10, classifier, human, meta, device)
    # Fit

    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    AB.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
           Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)  # 80

    test_data = AB.test(Dataset_CIFAR.data_test_loader, 10)
    out = cov_vs_acc_add(test_data)
    plot_cov_vs_acc([out], "AB", "Results/AB.pdf")


def test_learned_beyond_loss():
    # image
    dataset = CifarSynthDataset(5, False, batch_size=512)

    # models
    classifier, meta = networks("cifar_synth", "LearnedBeyond", device)

    # Additional
    Add = LearnedBeyond(10, classifier, meta, device)

    x, y, m = next(iter(dataset.data_train_loader))
    # make m one-hot
    m_oh = torch.nn.functional.one_hot(m, num_classes=10).float()
    x = x.to(device)
    y = y.to(device)
    m = m.to(device)
    m_oh = m_oh.to(device)
    assert m_oh.shape == torch.Size([512, 10])
    model_pred = classifier(x)
    meta_pred = meta(x, m_oh)

    loss = Add.surrogate_loss_bce(model_pred, meta_pred, y)
    print("loss: ", loss)
    assert isinstance(loss, torch.Tensor)
    assert not np.isnan(loss.detach().cpu().numpy()).any()
    assert loss.shape == torch.Size([])
    assert loss >= 0
    print("Test Learned Loss passed!")


def test_learned_beyond_fit():
    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)

    # Initialize method
    classifier, meta = networks("cifar_synth", "LearnedBeyond", device)

    LB = LearnedBeyond(10, classifier, meta, device)

    # Fit
    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    LB.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
           Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)  # 150
    test_data = LB.test(Dataset_CIFAR.data_test_loader, 10)
    res = cov_vs_acc_meta(test_data)
    plot_cov_vs_acc([res], "LB", "Results/LB.pdf")

    print("Test Additional fit passed!")


def test_learned_additional_fit():
    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)

    # Initialize method
    classifier, meta = networks("cifar_synth", "LearnedAdditional", device)

    LA = LearnedAdditional(10, classifier, meta, device)

    # Fit
    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    LA.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
           Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)  # 150
    test_data = LA.test(Dataset_CIFAR.data_test_loader, 10)
    res = cov_vs_acc_add(test_data)
    plot_cov_vs_acc([res], "LA", "Results/LA.pdf", std=False)
    print("Test Learned Additional fit passed!")


def test_beyond_fit_cifar10h():

    # Image
    Dataset_CIFAR = Cifar10h(False, data_dir='./data')

    # Initialize method
    classifier, human, meta = networks("cifar_10h", "BD", device, sp_epochs=1)

    B = BeyondDefer(10, classifier, human, meta, device)

    # Fit
    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    B.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
          Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
          scheduler=scheduler, verbose=True)  # 30
    test_data = B.test(Dataset_CIFAR.data_test_loader, 10)
    # plot histogram of test_data rej_score
    plt.figure()
    plt.hist(test_data["rej_score"], bins=100)
    plt.savefig("./BD_hist.pdf")
    res = cov_vs_acc_meta(test_data, method="c")
    accs = [m["system_acc"] for m in res]
    covs = [m["coverage"] for m in res]
    c = np.arange(0, 1, 1 / len(accs))
    loss = 1 - np.array(accs) + (1 - np.array(covs))*c
    acc_part = 1 - np.array(accs)
    cov_part = (1 - np.array(covs))*c
    Res = []
    for i in range(len(accs)):
        Res.append({"system_acc": accs[i], "c": c[i],
                    "coverage": covs[i], "loss": loss[i]})
    plot_cov_vs_acc([Res], ["BD"], "BD.pdf", method="c",
                    is_loss=True)
    plt.figure()
    plt.plot(c, acc_part, label="acc part")
    plt.plot(c, cov_part, label="cov part")
    plt.plot(c, np.array(covs), label="cov")
    plt.legend()
    plt.savefig("./BD_parts.pdf")
    print("Test Beyond fit passed!")


def test_additional_cost():

    # Image
    Dataset_CIFAR = CifarSynthDataset(5, False, batch_size=512)

    # Initialize method
    classifier, human, meta = networks("cifar_synth", "Additional", device)

    # loss
    loss_matrix = torch.ones(10, 10) - torch.eye(10)
    # loss_matrix = torch.rand(10, 10)
    AC = AdditionalCost(10, classifier, human, meta, device, loss_matrix)

    # Fit
    optimizer, scheduler = optimizer_scheduler()
    AC.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
           Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)  # 30
    test_data = AC.test(Dataset_CIFAR.data_test_loader, 10)
    res = cov_vs_acc_add(test_data, loss_matrix=loss_matrix)
    plot_cov_vs_cost([res], "AC", "Results/AC.pdf", std=False)
    print("Test Additional fit passed!")


def test_cost_sensitive_deferral():

    # Image
    Dataset_CIFAR = CifarSynthDataset(5, False, batch_size=512)

    # Initialize method
    classifier = networks("cifar_synth", "LCE", device)

    # loss
    loss_matrix = torch.ones(10, 10) - torch.eye(10)

    # Fit
    optimizer, scheduler = optimizer_scheduler()
    CCE = LceCost(1, 2, classifier, device)
    CCE.set_loss_matrix(loss_matrix)
    CCE.fit_hyperparam(
            Dataset_CIFAR.data_train_loader,
            Dataset_CIFAR.data_val_loader,
            Dataset_CIFAR.data_test_loader,
            epochs=1,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=0.001,
            verbose=True,
            test_interval=5,
        )  # 30
    test_data = CCE.test(Dataset_CIFAR.data_test_loader)
    res_LCE = compute_coverage_v_acc_curve(test_data, loss_matrix=loss_matrix)
    plot_cov_vs_cost([res_LCE], ["LCE"], "Results/LCE.pdf", std=False)
    logging.info("Results: {}".format(test_data))


def test_cost_sensitive_compareconf():

    # Image
    Dataset_CIFAR = CifarSynthDataset(5, False, batch_size=512)

    # Initialize method
    classifier, expert = networks("cifar_synth", "confidence", device)

    # loss
    loss_matrix = torch.ones(10, 10) - torch.eye(10)

    # Fit
    optimizer, scheduler = optimizer_scheduler()
    CCC = CompareConfCost(classifier, expert, device)
    CCC.set_loss_matrix(loss_matrix)
    CCC.fit(
        Dataset_CIFAR.data_train_loader,
        Dataset_CIFAR.data_val_loader,
        Dataset_CIFAR.data_test_loader,
        epochs=1,
        optimizer=optimizer,
        scheduler=scheduler,
        lr=0.001,
        verbose=False,
        test_interval=5,
    )  # 30
    test_data = CCC.test(Dataset_CIFAR.data_test_loader)
    res_CCC = compute_coverage_v_acc_curve(test_data, loss_matrix=loss_matrix)
    plot_cov_vs_cost([res_CCC], ["CCC"], "Results/CCC.pdf", std=False)
    logging.info("Test cost sensitive compareconf passed!")


def test_cost_sensitive_OvA():

    # Image
    Dataset_CIFAR = CifarSynthDataset(5, False, batch_size=512)

    # Initialize method
    model = networks("cifar_synth", "OVA", device)
    optimizer, scheduler = optimizer_scheduler()
    OVA = OVACost(1, 2, model, device)
    loss_matrix = torch.ones(10, 10) - torch.eye(10)
    OVA.set_loss_matrix(loss_matrix)
    OVA.fit(
            Dataset_CIFAR.data_train_loader,
            Dataset_CIFAR.data_val_loader,
            Dataset_CIFAR.data_test_loader,
            epochs=1,
            optimizer=optimizer,
            scheduler=scheduler,
            lr=0.001,
            verbose=False,
            test_interval=5,
        )  # 30
    test_data = OVA.test(Dataset_CIFAR.data_test_loader)
    res_OVAC = compute_coverage_v_acc_curve(test_data, loss_matrix=loss_matrix)
    plot_cov_vs_cost([res_OVAC], ["OVAC"], "Results/OVAC.pdf", std=False)
    logging.info("Test cost sensitive OvA passed!")


def test_CompConf_Meta_fit():

    # Image
    Dataset_CIFAR = CifarSynthDataset(5, False, batch_size=512)

    # Initialize method
    classifier, meta, defer, defer_meta = networks("cifar_synth",
                                                   "CompConfMeta", device)
    CCM = CompareConfMeta(10, classifier, meta, defer, defer_meta, device)

    # Fit
    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    CCM.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
            Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
            scheduler=scheduler, verbose=True)  # 30
    test_data = CCM.test(Dataset_CIFAR.data_test_loader, 10)
    res = cov_vs_acc_add(test_data)
    plot_cov_vs_acc([res], "CCM", "Results/test/CCM.pdf", std=False)

    print("Test CompConf_Meta_fit passed!")


def test_cost_sensitive_CompConf_Meta():

    # Image
    Dataset_CIFAR = CifarSynthDataset(5, False, batch_size=512)

    # Initialize method
    classifier, meta, defer, defer_meta = networks("cifar_synth",
                                                   "CompConfMeta", device)
    loss_matrix = torch.ones(10, 10) - torch.eye(10)
    CCMC = CompareConfMetaCost(10, classifier, meta, defer, defer_meta, device,
                               loss_matrix=loss_matrix)

    # Fit
    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    CCMC.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
             Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
             scheduler=scheduler, verbose=True)  # 30
    test_data = CCMC.test(Dataset_CIFAR.data_test_loader, 10)
    res = cov_vs_acc_add(test_data)
    plot_cov_vs_acc([res], "CCMC", "Results/test/CCMC.pdf", std=False)

    print("Test CompConf_Meta_Cost passed!")


def test_cifar_entropy_CCMC():

    # Image
    Dataset_CIFAR = CifarSynthDatasetEnt(10, False, batch_size=512)
    classifier, meta, defer, defer_meta = networks("cifar_synth",
                                                   "CompConfMeta", device)
    loss_matrix = torch.ones(10, 10) - torch.eye(10)
    CCMC = CompareConfMetaCost(10, classifier, meta, defer, defer_meta, device,
                               loss_matrix=loss_matrix)

    # Fit
    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    CCMC.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
             Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
             scheduler=scheduler, verbose=True)  # 30
    test_data = CCMC.test(Dataset_CIFAR.data_test_loader, 10)
    res = cov_vs_acc_add(test_data)
    plot_cov_vs_acc([res], "CCMC", "Results/test/Ent.pdf", std=False)
    logging.info("Test cifar entropy passed!")


def test_AFE_coverage():

    # Hate speech
    # Dataset_hate = HateSpeech("./data/", True, False, 'random_annotator',
    #                           device)
    Dataset_hate = ImageNet16h(False,
                               data_dir="./data/osfstorage-archive/",
                               noise_version="110",
                               batch_size=32,
                               test_split=0.2,
                               val_split=0.01)
    Dataset_hate_Active = ActiveDataset(Dataset_hate)
    # find data loader length
    length = len(Dataset_hate_Active.data_train_loader)
    # networks
    classifier, meta = networks("imagenet", "AFE", device)
    # AFE
    AFE_CIFAR = AFE(classifier, meta, device)
    optimizer, scheduler = optimizer_scheduler()
    # fit
    AFE_CIFAR.fit(Dataset_hate_Active,
                  16, 1, lr=0.001, verbose=True,
                  query_size=int(np.floor(length/1)),
                  num_queries=1, scheduler_classifier=scheduler,
                  scheduler_meta=scheduler)  # num_queries=10
    #  test
    plt.figure()
    range_epochs = np.arange(0, len(AFE_CIFAR.report))
    accuracies = []

    last_iter = len(AFE_CIFAR.report) - 1
    rep = AFE_CIFAR.report[last_iter]
    res = cov_vs_acc_AFE(rep)
    plot_cov_vs_acc([res], ["AFE"], "Results/AFE_cov.pdf", std=False)

    for i in range_epochs:
        accuracies.append(AFE_CIFAR.report[i]
                          ["test_metrics_meta"]["system_acc"])
    plt.plot(range_epochs, accuracies)
    plt.savefig("Results/AFE_CIFAR_system_acc.png")


def test_set_optimization():

    def function(x):
        if len(x) == 0:
            return 0
        else:
            x = np.array(x)
            length = len(x)
            if length < 5:
                t = 0
            else:
                t = length - 5
            return -np.sum(x) + t*10000

    # Test data
    data = []
    for i in range(100):
        rand = np.random.rand(1)
        data.append(rand)

    # Actual optimization
    data_np = np.array(data)
    data_np = np.sort(data_np, axis=0)
    data_opt = data_np[-5:]
    func_opt = function(data_opt)

    # Greedy optimization
    MySetFunc = SetFunction(function, data)
    res = MySetFunc.min([], iter=1000)
    logging.info("Result: {}".format(res))
    logging.info("data opt: {}".format(data_opt))
    logging.info("Optimal: {}".format(func_opt))
    assert res[0] == func_opt


def test_all_sizes():

    # CIFAR-10K
    Dataset_CIFAR = CifarSynthDataset(10, False, batch_size=512)

    # CIFAR-10H
    Dataset_CIFAR_H = Cifar10h(False, data_dir='./data')

    # ImageNet-16H
    Dataset_Imagenet = ImageNet16h(False,
                                   data_dir="./data/osfstorage-archive/",
                                   noise_version="110",
                                   batch_size=32,
                                   test_split=0.2,
                                   val_split=0.01)

    # Hate speech
    Dataset_hate = HateSpeech("./data/", True, False, 'random_annotator',
                              device)

    all_datasets = [Dataset_CIFAR, Dataset_CIFAR_H, Dataset_Imagenet,
                    Dataset_hate]

    name_datasets = ["CIFAR-10K", "CIFAR-10H", "ImageNet-16H", "Hate Speech"]

    for i, dataset in enumerate(all_datasets):
        train_num = len(dataset.data_train_loader.dataset)
        val_num = len(dataset.data_val_loader.dataset)
        test_num = len(dataset.data_test_loader.dataset)
        logging.info("Dataset: {}".format(name_datasets[i]))
        logging.info("Train: {}".format(train_num))
        logging.info("Val: {}".format(val_num))
        logging.info("Test: {}".format(test_num))

    logging.info("Test all sizes passed!")


def test_hist_rej1_rej2():
    Dataset_CIFAR = Cifar10h(False, data_dir='./data')

    # Initialize method
    classifier, meta, defer, defer_meta = \
        networks("cifar_10h", "CompConfMeta", device, sp_epochs=1)
    CCM = CompareConfMeta(10, classifier, meta, defer, defer_meta, device)

    # Fit
    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    CCM.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
            Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
            scheduler=scheduler, verbose=True)  # 50
    test_data = CCM.test(Dataset_CIFAR.data_test_loader, 10)
    plt.figure()
    plt.hist(test_data["rej_score2"], bins=100)
    plt.savefig("./Results/CCM_hist1.pdf")
    plt.figure()
    plt.hist(test_data["rej_score2"]-test_data["rej_score1"], bins=100)
    plt.savefig("./Results/CCM_hist2.pdf")
    logging.info("Test hist rej1 rej2 passed!")


def test_CCM_ABD_Simplex():
    Dataset_CIFAR = Cifar10h(False, data_dir='./data')

    # Initialize method
    classifier, meta, defer, defer_meta = \
        networks("cifar_10h", "CompConfMeta", device, sp_epochs=1)
    CompareConfMeta(10, classifier, meta, defer, defer_meta, device)

    classifier_AB, human_AB, meta_AB = \
        networks("cifar_10h", "Additional", device, sp_epochs=1)
    AB = AdditionalBeyond(10, classifier_AB, human_AB, meta_AB, device)

    # Fit
    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    # CCM.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
    #         Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.001,
    #         scheduler=scheduler, verbose=True)
    # test_data = CCM.test(Dataset_CIFAR.data_test_loader, 10)
    # class_probs = test_data["class_probs"]
    # class_probs = np.sum(class_probs, axis=1)
    
    AB.fit(Dataset_CIFAR.data_train_loader, Dataset_CIFAR.data_val_loader,
           Dataset_CIFAR.data_test_loader, 10, 1, optimizer, lr=0.01,
           scheduler=scheduler, verbose=True)  # 50
    test_data_AB = AB.test(Dataset_CIFAR.data_test_loader, 10)
    class_probs_AB = test_data_AB["class_probs"]
    class_probs_AB = np.sum(class_probs_AB, axis=1)
    plt.figure()
    plt.hist(class_probs_AB, bins=100)
    # plt.hist(class_probs, bins=1)
    # logging.info("class probs: {}".format(class_probs))
    # logging.info("class probs AB: {}".format(class_probs_AB))
    plt.savefig("./Results/CCM_ABD_Simplex.pdf")
    logging.info("Test CCM ABD Simplex passed!")


def test_AFE_imagenet():
    Dataset_Imagenet = ImageNet16h(False,
                                   data_dir="./data/osfstorage-archive/",
                                   noise_version="110",
                                   batch_size=32,
                                   test_split=0.2,
                                   val_split=0.01)
    Dataset_Active = ActiveDataset(Dataset_Imagenet)

    # Initialize method
    classifier, meta = networks("imagenet", "AFE", device)
    AFE_Image = AFE(classifier, meta, device)

    # Fit
    optimizer, scheduler = optimizer_scheduler()
    length = len(Dataset_Imagenet.data_test_loader.dataset)
    AFE_Image.fit(Dataset_Active,
                  16, 1, lr=0.001, verbose=True,
                  query_size=int(np.floor(length)),
                  num_queries=1, scheduler_classifier=scheduler,
                  scheduler_meta=scheduler, optimizer=optimizer)  # 10

    plt.figure()
    range_epochs = np.arange(0, len(AFE_Image.report))
    accuracies = []

    last_iter = len(AFE_Image.report) - 1
    rep = AFE_Image.report[last_iter]
    res = cov_vs_acc_AFE(rep)
    plot_cov_vs_acc([res], ["AFE"], "Results/AFE_ImageNet_cov.pdf", std=False)

    for i in range_epochs:
        accuracies.append(AFE_Image.report[i]
                          ["test_metrics_meta"]["system_acc"])
    plt.plot(range_epochs, accuracies)
    plt.savefig("Results/AFE_ImageNet_system_acc.png")


def test_all():
    # test_indexed()
    # test_active_mask()
    # test_active_query()
    # test_Meta_model()
    # test_AFE_loss()
    # test_AFE_loss_loaders()
    # test_AFE_CE_loss()
    # test_AFE_fit_epochs()
    # test_AFE_fit_Eu()
    # test_Query_unnumbered()
    # test_Query_test()
    # test_iteration_report()
    # test_AFE_fit()
    # test_OVA_loss()
    # test_BD_loss()
    # test_BD_fit_epoch()
    # test_BD_fit()
    test_BD_fit_CIFAR10h()
    test_BD_fit_Imagenet()
    test_RS_Imagenet()
    test_BD_Hatespeech()
    test_RS_Hatespeech()
    test_parallel()
    test_return_res()
    test_compute_meta()
    test_additional_defer_loss()
    test_additional_defer_fit()
    test_BCE_loss_vs_OvA()
    test_cov_vs_acc()
    test_learned_beyond_loss()
    test_learned_beyond_fit()
    test_learned_additional_fit()
    test_beyond_fit_cifar10h()
    test_additional_cost()
    test_cost_sensitive_deferral()
    test_cost_sensitive_compareconf()
    test_cost_sensitive_OvA()
    test_CompConf_Meta_fit()
    test_cost_sensitive_CompConf_Meta()
    test_cifar_entropy_CCMC()
    test_AFE_coverage()
    test_set_optimization()
    test_all_sizes()
    test_hist_rej1_rej2()
    test_CCM_ABD_Simplex()
    test_AFE_imagenet()
    logging.info("All tests passed!")
