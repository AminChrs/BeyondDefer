# In this file, I am finding a thresholding over the D(P(Y|X), P(Y|X, M))
# Based on which I decide whether to collect the human feature or not
# Afterwards, I update my belief about the human label and re-train the model
# The threshold is found via a grid search in validation set

import sys
import torch
sys.path.append("..", 0)
from human_ai_deferral.baselines.basemethod import BaseMethod
from human_ai_deferral.helpers.utils import *
import logging

class AFE(BaseMethod):
    def __init__(self, model_classifier, model_meta, device):
        self.model_classifier = model_classifier
        self.model_meta = model_meta
        self.device = device

    def AFELoss(self, classifier_pred, meta_pred):
        # find the KL divergence between the two distributions
        return torch.nn.functional.kl_div(classifier_pred, meta_pred)

    def Loss(self, pred, y):
        return torch.nn.CrossEntropyLoss(pred, y)

    def fit_Eo_epoch(self, dataloader, n_classes, optimizer, verbose = False, epoch = 1):
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

        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
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
                logging.warning(f"NAN LOSS")
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
