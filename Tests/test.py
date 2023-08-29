import sys
sys.path.append("../")
from Feature_Acquisition.active import IndexedDataset, ActiveDataset, AFE
from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_deferral.datasetsdefer.hatespeech import HateSpeech
from MyNet.networks import MetaNet
from human_ai_deferral.networks.cnn import NetSimple
import torch
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print("Test Indexed Dataset Passed!")


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
    print("Test Active Dataset Passed!")


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
    print("Test Active Query Passed!")


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
    print("Test AFE Passed!")


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
    print("Test AFE Loss Loaders Passed!")


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
    print("Test Meta Model Passed!")


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

    print("Test AFE Fit Classifier Passed!")


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
    print("Test AFE CE Loss Passed!")


def test_AFE_fit_Eu():

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

    optimizer_meta = torch.optim.Adam(Meta.parameters(), lr=0.001)

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
    Dataset_CIFAR_Active.Query(criterion, pool_size=0, query_size=1)
    AFE_CIFAR.fit_Eu(2, Dataset_CIFAR_Active,
                     10, optimizer_meta, verbose=True)
    print("Test AFE Fit Eu Passed!")


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

    AFE_CIFAR.fit(Dataset_CIFAR_Active,
                  10, 1, lr=0.001, verbose=True, query_size=10)
    print("Test AFE fit passed!")


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
    print("Test Query Unnumbered passed!")

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
    print(loss)
    assert isinstance(loss, float)
    assert loss >= 0
    assert not np.isnan(loss)
    print("Test Query Test passed!")


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

# test_indexed()
# test_active_mask()
# test_active_query()
# test_Meta_model()
# test_AFE_loss()
# test_AFE_loss_loaders()
# test_AFE_CE_loss()
# test_AFE_fit_epochs()
# test_AFE_fit_Eu()
test_AFE_fit()
# test_Query_unnumbered()
# test_Query_test()
# test_iteration_report()