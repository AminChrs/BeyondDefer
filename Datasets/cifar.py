import torch
from human_ai_defer.datasetsdefer.basedataset import BaseDataset
from human_ai_defer.datasetsdefer.generic_dataset import\
     GenericImageExpertDataset
import numpy as np
import random
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


class CifarSynthExpertEnt:
    """simple class to describe our synthetic expert on CIFAR-10
    k: number of classes expert can predict, n_classes: number of classes
    (10 for CIFAR-10)"""

    def __init__(self, k, n_classes):
        self.k = k
        self.n_classes = n_classes

    def predict(self, labels):
        batch_size = len(labels)
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i] <= self.k - 1:
                rand = random.uniform(0, 1)
                if rand < (labels[i] * 10 + 10) / 100:
                    prediction_rand = labels[i]
                else:
                    prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs


class CifarSynthDatasetEnt(BaseDataset):
    """This is the Cifar Dataset for checking the conditional entropy of
    the labels given expert and the covariates"""

    def __init__(
        self,
        expert_k,
        use_data_aug,
        test_split=0.2,
        val_split=0.1,
        batch_size=1000,
        n_dataset=10,
        transforms=None,
    ):
        """
        expert_k: number of classes expert can predict
        use_data_aug: whether to use data augmentation (bool)
        test_split: NOT USED FOR CIFAR, since we have a fixed test set
        val_split: percentage of data to be used for validation
        batch_size: batch size for training
        transforms: data transforms
        """
        self.expert_k = expert_k
        self.use_data_aug = use_data_aug
        self.n_dataset = n_dataset
        self.expert_fn = CifarSynthExpertEnt(expert_k, self.n_dataset).predict
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def generate_data(self):
        """
        generate data for training, validation and test sets
        """
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        if self.use_data_aug:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: F.pad(
                            x.unsqueeze(0), (4, 4, 4, 4), mode="reflect"
                        ).squeeze()
                    ),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if self.n_dataset == 10:
            dataset = "cifar10"
        elif self.n_dataset == 100:
            dataset = "cifar100"

        train_dataset_all = datasets.__dict__[dataset.upper()](
            "../data", train=True, download=True, transform=transform_train
        )
        train_size = int((1 - self.val_split) * len(train_dataset_all))
        val_size = len(train_dataset_all) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset_all, [train_size, val_size]
        )

        test_dataset = datasets.__dict__["cifar10".upper()](
            "../data", train=False, transform=transform_test, download=True
        )

        dataset_train = GenericImageExpertDataset(
            np.array(train_dataset.dataset.data)[train_dataset.indices],
            np.array(train_dataset.dataset.targets)[train_dataset.indices],
            self.expert_fn(
                np.array(train_dataset.dataset.targets)[train_dataset.indices]
            ),
            transform_train,
        )
        dataset_val = GenericImageExpertDataset(
            np.array(val_dataset.dataset.data)[val_dataset.indices],
            np.array(val_dataset.dataset.targets)[val_dataset.indices],
            self.expert_fn(np.array(val_dataset.dataset.targets
                                    )[val_dataset.indices]),
            transform_test,
        )
        dataset_test = GenericImageExpertDataset(
            test_dataset.data,
            test_dataset.targets,
            self.expert_fn(test_dataset.targets),
            transform_test,
        )

        self.data_train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.data_val_loader = DataLoader(
            dataset=dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        self.data_test_loader = DataLoader(
            dataset=dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
