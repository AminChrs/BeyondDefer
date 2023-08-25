import sys
sys.path.append("../")
from Feature_Acquisition.active import IndexedDataset, ActiveDataset, AFE
from human_ai_deferral.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_deferral.datasetsdefer.hatespeech import HateSpeech
from MyNet.networks import MetaNet
from human_ai_deferral.networks.cnn import NetSimple
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    dataset_actual = Dataset_CIFAR_Active.train_dataset
    # Sample Randomly from the dataset to make a loader
    indices = np.random.choice(len(dataset_actual), 100, replace=False)
    inv_indices = np.setdiff1d(np.arange(len(dataset_actual)), indices)
    dataset_loader1 = torch.utils.data.DataLoader(
        dataset_actual, batch_size=1,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
    dataset_loader2 = torch.utils.data.DataLoader(
        dataset_actual, batch_size=1,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(inv_indices))

    # Classifier
    Classifier = NetSimple(10, 50, 50, 100, 20).to(device)
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)

    # AFE
    AFE_CIFAR = AFE(Classifier, Meta, device)
    print(AFE_CIFAR.AFELoss_loaders(dataset_loader1, dataset_loader2, 10))
    assert AFE_CIFAR.AFELoss_loaders(dataset_loader1, dataset_loader2, 10).shape == \
        torch.Size([])


def test_Meta_model():

    # Image
    expert_k = 5
    Dataset_CIFAR = CifarSynthDataset(expert_k, False, batch_size=512)
    train_loader = Dataset_CIFAR.data_train_loader

    # Classifier
    Meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20), [1, 20, 1],
                   remove_layers=["fc3", "softmax"]).to(device)
    print(Meta)

    for batch, (x, y, m) in enumerate(train_loader):
        x = x.to(device)
        m = m.to(device)
        assert x.shape == torch.Size([512, 3, 32, 32])
        assert y.shape == torch.Size([512])
        # make m a 1-hot vector
        m = torch.nn.functional.one_hot(m, 10).float()
        assert m.shape == torch.Size([512, 10])

        assert Meta(x, m).shape == torch.Size([512, 10])
        break


# test_indexed()
# test_active_mask()
# test_active_query()
# test_Meta_model()
test_AFE_loss()
test_AFE_loss_loaders()
