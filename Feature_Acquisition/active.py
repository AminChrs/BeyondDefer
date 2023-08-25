# In this file, I am finding a thresholding over the D(P(Y|X), P(Y|X, M))
# Based on which I decide whether to collect the human feature or not
# Afterwards, I update my belief about the human label and re-train the model
# The threshold is found via a grid search in validation set
import sys
sys.path.append("..")
sys.path.append("../human_ai_deferral")
import copy
import torch
from human_ai_deferral.datasetsdefer.basedataset import BaseDataset
from human_ai_deferral.baselines.basemethod import BaseMethod
from human_ai_deferral.helpers.utils import AverageMeter, accuracy
from human_ai_deferral.helpers.metrics import compute_metalearner_metrics
from human_ai_deferral.helpers.metrics import compute_classification_metrics
import logging
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
import time
import numpy as np


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset.data_train_loader.dataset

    def __getitem__(self, index):
        return index, self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)


class ActiveDataset(BaseDataset):
    def __init__(self, original_dataset):
        self.__dict__ = original_dataset.__dict__.copy()
        self.train_dataset = IndexedDataset(original_dataset)
        self.mask_labeled = torch.zeros(len(self.train_dataset.dataset))
        self.labeled_loader = None

    def mask(self, idx):
        self.mask_labeled[idx] = 1

    def mask_label(self, idx):
        return self.mask_labeled[idx]

    def generate_data(self):
        return super().generate_data()

    def Query(self, criterion, pool_size=100, batch_size=128, query_size=10,
              verbose=False):
        unlabeled_idx = torch.where(self.mask_labeled == 0)[0]
        if (pool_size > 0):
            pool_idx = random.sample(range(1, len(unlabeled_idx)), pool_size)
        else:
            pool_idx = np.arange(len(unlabeled_idx))
        pool_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                 sampler=SubsetRandomSampler(
                                    unlabeled_idx[pool_idx]))
        labeled_idx = torch.where(self.mask_labeled == 1)[0]
        labeled_loader = DataLoader(self.train_dataset,
                                    batch_size=batch_size,
                                    sampler=SubsetRandomSampler(
                                        labeled_idx))

        loss, indices = criterion(pool_loader, labeled_loader)

        loss, indices_ordered = torch.sort(loss, descending=True)
        indices_query = indices[indices_ordered[:query_size]]
        if (indices_query.shape == torch.Size([])):
            indices_query = torch.tensor([indices_query])
        for i in range(len(indices_query)):
            self.mask(indices_query[i])
        labeled_idx = torch.where(self.mask_labeled == 1)[0]
        self.labeled_loader = DataLoader(self.train_dataset,
                                         batch_size=batch_size,
                                         sampler=SubsetRandomSampler(
                                                        labeled_idx))


class AFE(BaseMethod):
    def __init__(self, model_classifier, model_meta, device):
        self.model_classifier = model_classifier
        self.model_meta = model_meta
        self.device = device

    def AFELoss(self, classifier_pred, meta_pred):
        classifier_pred = torch.nn.functional.softmax(classifier_pred, dim=1)
        meta_pred = torch.nn.functional.softmax(meta_pred, dim=1)
        return torch.sum(classifier_pred * torch.log(classifier_pred /
                                                     meta_pred))

    def AFELoss_loaders(self, dataloader_labeled, dataloader_unlabeled,
                        num_classes):
        """ Compute the AFE loss for the whole dataset
        
        dataloader_labeled: dataloader for labeled data
        dataloader_unlabeled: dataloader for unlabeled data
        """
        KL_loss = []
        normalizer = []
        indices = []
        batch = -1
        for iters in enumerate(dataloader_unlabeled):
            if len(iters) == 2:
                batch = iters[0]
                indices, (data_x, data_y, _) = iters[1]
            else:
                data_x, data_y, _ = iters
                batch += 1
            batch_size = data_x.size()[0]
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            output_classifier = self.model_classifier(data_x)
            for iters_in in enumerate(dataloader_labeled):
                if len(iters_in) == 2:
                    batch_l, (indices_l, (data_x_l, _, hum_preds)) =\
                         iters_in
                else:
                    data_x_l, _, hum_preds = iters_in
                    batch_l += 1
                batch_size_l = data_x.size()[0]
                data_x_l = data_x.to(self.device)
                hum_preds = hum_preds.to(self.device)
                # check whether human predictions are one-hot encoded
                if len(hum_preds.size()) == batch_l.shape:
                    hum_preds = torch.nn.functional.one_hot(hum_preds,
                                                            num_classes)
                output_meta = self.model_meta(data_x_l, hum_preds)

                for i in range(batch_size):
                    for j in range(batch_size_l):
                        if j == 0:
                            KL_loss.append(self.AFELoss(output_classifier[i],
                                                        output_meta[j]))
                            normalizer.append(1)
                            indices.append(indices[i])
                        else:
                            KL_loss[i] += self.AFELoss(output_classifier[i],
                                                       output_meta[j])
                            normalizer[i] += 1
        KL_loss /= normalizer
        return KL_loss, indices

    def Loss(self, pred, y):
        return torch.nn.CrossEntropyLoss(pred, y)

    def fit_Eo_epoch(self, dataloader, n_classes, optimizer,
                     verbose=False, epoch=1):
        """ Fit the meta model for one epoch and on labeled data only
        model: meta model
        dataloader: dataloader for labeled data
        optimizer: optimizer for meta model
        verbose: print loss
        epoch: current epoch
        """

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        
        self.model_meta.train()

        for batch, (idx, data_x, data_y, hum_preds) in enumerate(dataloader):

            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            one_hot_m = torch.zeros((data_x.size()[0], n_classes))
            one_hot_m[torch.arange(data_x.size()[0]), hum_preds] = 1
            one_hot_m = one_hot_m.to(self.device)

            output_meta = self.model_meta(data_x, one_hot_m)
            loss = self.Loss(output_meta, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            prec = accuracy(output_meta.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec.item(), data_x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning("NAN LOSS")
                break
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 Meta {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )

    def fit_El_epoch(self, dataloader, n_classes, optimizer,
                     verbose=False, epoch=1):
        """ Fit the classifier model on all data
        model: classifier model
        dataloader: dataloader for all data
        optimizer: optimizer for classifier model
        verbose: print loss
        epoch: current epoch
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        self.model_classifier.train()

        for batch, (idx, data_x, data_y, _) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)

            output_classifier = self.model_classifier(data_x)
            loss = self.Loss(output_classifier, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            prec = accuracy(output_classifier.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec.item(), data_x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning("NAN LOSS")
                break
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 Classifier {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )

    def fit_Eu(self, epochs, train_loader, test_loader, n_classes,
               optimizer_meta, verbose=False, test_interval=5,
               scheduler_meta=None):
        best_acc = 0
        best_model_meta = None
        for epoch in range(epochs):
            self.fit_Eu_epoch(train_loader,
                              n_classes,
                              optimizer_meta,
                              verbose=verbose,
                              epoch=epoch)

            if epoch % test_interval == 0:
                data_test = self.test(test_loader, n_classes)
                val_metrics = compute_classification_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:
                    best_acc = val_metrics["system_acc"]
                    best_model_meta = copy.deepcopy(
                                self.model_classifier.state_dict())
                if verbose:
                    logging.info(compute_metalearner_metrics(data_test))
            if scheduler_meta is not None:
                scheduler_meta.step()
            self.model_meta.load_state_dict(best_model_meta)

    def fit(self,
            train_loader,
            val_loader,
            test_loader,
            n_classes,
            optimizer_classifier,
            optimizer_meta,
            epochs,
            lr,
            scheduler_classifier=None,
            scheduler_meta=None,
            verbose=False,
            test_interval=5):

        """Fit the classifier, then find the KL divergence between each
         unlabeled points and all the labeled ones, and pick the highest
         one, and then collect the human feature for that point, and
         re-train the classifier.

        train_loader: dataloader for labeled data
        val_loader: dataloader for unlabeled data
        test_loader: dataloader for test data
        n_classes: number of classes
        optimizer_classifier: optimizer for classifier model
        optimizer_meta: optimizer for meta model
        epochs: number of epochs
        lr: learning rate
        scheduler_classifier: scheduler for classifier model
        scheduler_meta: scheduler for meta model
        verbose: print loss
        """

        params_class = list(self.model_classifier.parameters())
        params_meta = list(self.model_meta.parameters())
        optimizer_classifier = optimizer_classifier(params_class, lr=lr)
        optimizer_meta = optimizer_meta(params_meta, lr=lr)

        if scheduler_classifier is not None:
            scheduler_classifier = scheduler_classifier(optimizer_classifier)
        if scheduler_meta is not None:
            scheduler_meta = scheduler_meta(optimizer_meta)

        best_acc = 0
        best_model_class = copy.deepcopy(self.model_classifier.state_dict())
        # best_model_meta = copy.deepcopy(self.model_meta.state_dict())
        for epoch in range(epochs):
            self.fit_El_epoch(train_loader,
                              n_classes,
                              optimizer_classifier,
                              verbose=verbose,
                              epoch=epoch)

            if epoch % test_interval == 0:
                data_test = self.test(val_loader, n_classes)
                val_metrics = compute_classification_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:
                    best_acc = val_metrics["system_acc"]
                    best_model_class = copy.deepcopy(
                                self.model_classifier.state_dict())
                if verbose:
                    logging.info(compute_classification_metrics(data_test))
            if scheduler_classifier is not None:
                scheduler_classifier.step()
        self.model_classifier.load_state_dict(best_model_class)

        # Next, I make a new dataloader that I add a labeled data after 
        # acquisition. Then, I find the KL divergence between each unlabeled
        # and the labeled data, and pick the highest one and add it to the
        # labeled data. Then, I re-train the meta model on the new data.

        # First choose one point from the train loader and make a new 
        # dataloader with that one point
        loader_lu = ActiveDataset(train_loader)

        # Query
        query_size = 1
        all_queries = 100
        loader_lu.query(self.AFELoss_loaders, pool_size=0,
                        query_size=query_size)

        # train the meta model on the new data
        def Fit_Unlabeled():  self.fit_Eu(epochs, loader_lu, test_loader,
                                          n_classes,
                                          optimizer_meta,
                                          verbose=verbose,
                                          test_interval=test_interval,
                                          scheduler_meta=scheduler_meta)

        Fit_Unlabeled()

        for i in range(all_queries):
            loader_lu.Query(self.AFELoss_loaders, pool_size=0,
                            query_size=query_size)
            Fit_Unlabeled()

    def test():
        return super().test()