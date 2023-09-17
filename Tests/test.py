import sys
sys.path.append("../")
from Feature_Acquisition.active import IndexedDataset, ActiveDataset, AFE
from MyNet.call_net import networks, optimizer_scheduler
from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_deferral.datasetsdefer.hatespeech import HateSpeech
from human_ai_deferral.datasetsdefer.cifar_h import Cifar10h
from human_ai_deferral.datasetsdefer.imagenet_16h import ImageNet16h
from MyNet.networks import MetaNet
from MyMethod.beyond_defer import BeyondDefer
from human_ai_deferral.networks.cnn import NetSimple
import torch
import numpy as np
from torch.utils.data import DataLoader
import json
import logging
import matplotlib.pyplot as plt

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
        data_dir = '../human_ai_deferral/data/'
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
        data_dir = '../human_ai_deferral/data/'
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
            torch.Size([])
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
    AFE_CIFAR.fit_Eu(150, Dataset_CIFAR_Active,
                     10, optimizer_meta, verbose=True,
                     scheduler_meta=scheduler)
    logging.info("Test AFE Fit Eu Passed!")


def test_AFE_fit():

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

    def scheduler(z):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, 34000*50)

    AFE_CIFAR.fit(Dataset_CIFAR_Active,
                  10, 50, lr=0.001, verbose=True, query_size=100,
                  num_queries=100, scheduler_classifier=scheduler,
                  scheduler_meta=scheduler)
    # save AFE_CIFAR.report as a Json file
    with open('AFE_CIFAR_report.json', 'w') as fp:
        json.dump(AFE_CIFAR.report, fp)
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

    loss = Dataset_CIFAR_Active.Query_test(criterion, AFE_CIFAR.loss_defer, 10)
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
           Dataset_CIFAR.data_test_loader, 10, 80, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)
    print("Test BD fit passed!")


def test_BD_fit_CIFAR10h():

    # image
    dataset = Cifar10h(False, data_dir='../data')

    # models
    classifier, human, meta = networks("cifar_10h", "BD", device)

    # BD
    BD = BeyondDefer(10, classifier, human, meta, device)
    optimizer, scheduler = optimizer_scheduler()
    # fit
    BD.fit(dataset.data_train_loader, dataset.data_val_loader,
           dataset.data_test_loader, 10, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)
    print("Test BD on CIFAR-10H fit passed!")

def test_BD_fit_imagenet():

    # Image 
    dataset = ImageNet16h(False, data_dir="../data/osfstorage-archive/",
                          noise_version="125", batch_size=32, test_split=0.2,
                          val_split=0.01)

    # models
    classifier, human, meta = networks("imagenet_16h", "BD", device)

    # BD
    BD = BeyondDefer(16, classifier, human, meta, device)
    optimizer, scheduler = optimizer_scheduler()
    # fit
    BD.fit(dataset.data_train_loader, dataset.data_val_loader,
           dataset.data_test_loader, 10, 1, optimizer, lr=0.001,
           scheduler=scheduler, verbose=True)
    print("Test BD on ImageNet-16H fit passed!")

if __name__ == "__main__":
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
    # test_BD_fit_CIFAR10h()
    test_BD_fit_imagenet()
