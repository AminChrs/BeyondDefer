import copy
import torch
import numpy as np
import torch.nn.functional as F
import time
import logging
from tqdm import tqdm
from human_ai_defer.helpers.utils import AverageMeter, accuracy
from metrics.metrics import compute_additional_defer_metrics
from human_ai_defer.baselines.basemethod import BaseMethod

eps_cst = 1e-8
#  Additional Learned + CompareConf


class CompareConfMeta(BaseMethod):
    def __init__(self, plotting_interval, model_classifier,
                 model_meta, model_defer, model_defer_meta, device,
                 learnable_threshold_rej=False):
        '''
        plotting_interval (int): used for plotting model training in fit_epoch
        model_classifier (pytorch model): model used for surrogate
        device: cuda device or cpu
        learnable_threshold_rej (bool): whether to learn a treshold on the
        '''
        self.plotting_interval = plotting_interval
        self.model_classifier = model_classifier
        self.model_meta = model_meta
        self.model_defer = model_defer
        self.model_defer_meta = model_defer_meta
        self.device = device
        self.threshold_rej = 0
        self.learnable_threshold_rej = learnable_threshold_rej

    def LossBCEH(self, outputs, y, outputs_meta, outputs_human):
        num_classes = outputs.size()[1]
        y_oh = F.one_hot(y, num_classes=num_classes).float()
        m = torch.argmax(outputs_meta, dim=1).float()
        y_oh[:, -1] = (m == y).float()
        m2 = outputs_human
        y_oh[:, -2] = (m2 == y).float()
        return F.binary_cross_entropy_with_logits(outputs, y_oh,
                                                  reduction='none').sum(dim=1)

    def LossBCE(self, outputs, y):
        num_classes = outputs.size()[1]
        y_oh = F.one_hot(y, num_classes=num_classes).float()
        return F.binary_cross_entropy_with_logits(outputs, y_oh,
                                                  reduction='none').sum(dim=1)

    def surrogate_loss_bce(self, out_class, outputs_meta,
                           data_y, outputs_human):
        """
        outputs: network outputs
        m: cost of deferring to expert cost of classifier predicting
         hum_preds == target
        labels: target
        """
        return (self.LossBCEH(out_class, data_y, outputs_meta, outputs_human)
                + self.LossBCE(outputs_meta, data_y)).mean()

    def fit_epoch(self, dataloader, n_classes, optimizer,
                  verbose=False, epoch=1, model="classifier"):
        """
        Fit the model for one epoch
        model: model to be trained
        dataloader: dataloader
        optimizer: optimizer
        verbose: print loss
        epoch: epoch number
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1_classifier = AverageMeter()
        top1_meta = AverageMeter()
        end = time.time()
        self.model_classifier.train()
        self.model_meta.train()

        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            one_hot_m = torch.zeros((data_x.size()[0], n_classes))
            one_hot_m[torch.arange(data_x.size()[0]), hum_preds] = 1
            one_hot_m = one_hot_m.to(self.device)
            criterion = torch.nn.CrossEntropyLoss()
            if model == "classifier":
                outputs_classifier = self.model_classifier(data_x)
                loss = criterion(outputs_classifier, data_y)
            elif model == "meta":
                outputs_meta = self.model_meta(data_x, one_hot_m)
                loss = criterion(outputs_meta, data_y)
            elif model == "defer_meta":
                outputs_meta = self.model_meta(data_x, one_hot_m)
                meta_correct = torch.argmax(outputs_meta, dim=1) == data_y
                meta_correct = meta_correct.long().detach()
                outputs_defer_meta = self.model_defer_meta(data_x)
                loss = criterion(outputs_defer_meta, meta_correct)
            elif model == "defer":
                human_correct = hum_preds == data_y
                human_correct = human_correct.long().detach()
                outputs_defer = self.model_defer(data_x)
                loss = criterion(outputs_defer, human_correct)
            outputs_classifier = self.model_classifier(data_x)
            outputs_meta = self.model_meta(data_x, one_hot_m)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1_classifier = accuracy(outputs_classifier.data[:, :-2],
                                        data_y, topk=(1,))[0]
            prec1_meta = accuracy(outputs_meta.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1_classifier.update(prec1_classifier.item(), data_x.size(0))
            top1_meta.update(prec1_meta.item(), data_x.size(0))

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
                    "Prec@1 Classifier {top1_classifier.val:.3f} \
                                ({top1_classifier.avg:.3f})\t"
                    "Prec@1 Meta {top1_meta.val:.3f} \
                                    ({top1_meta.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1_classifier=top1_classifier,
                        top1_meta=top1_meta,
                    )
                )

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        n_classes,
        epochs,
        optimizer,
        lr,
        scheduler=None,
        verbose=True,
        test_interval=1,
    ):
        params = list(self.model_classifier.parameters()) + \
                        list(self.model_meta.parameters()) + \
                        list(self.model_defer.parameters()) + \
                        list(self.model_defer_meta.parameters())
        optimizer = optimizer(params, lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
        best_acc = 0

        # store current model dict
        best_model_classifier = copy.deepcopy(
                                self.model_classifier.state_dict())
        best_model_meta = copy.deepcopy(self.model_meta.state_dict())
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, n_classes, optimizer, verbose,
                           epoch, model="classifier")
            self.fit_epoch(dataloader_train, n_classes, optimizer, verbose,
                           epoch, model="meta")
            self.fit_epoch(dataloader_train, n_classes, optimizer, verbose,
                           epoch, model="defer_meta")
            self.fit_epoch(dataloader_train, n_classes, optimizer, verbose,
                           epoch, model="defer")

            if epoch % test_interval == 0 and epoch > 1:
                data_test = self.test(dataloader_val, n_classes)
                val_metrics = compute_additional_defer_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:
                    logging.info("New best model")
                    best_acc = val_metrics["system_acc"]
                    best_model_classifier = copy.deepcopy(
                                    self.model_classifier.state_dict())
                    best_model_meta = copy.deepcopy(
                                    self.model_meta.state_dict())
                if verbose:
                    logging.info(compute_additional_defer_metrics(data_test))
            if scheduler is not None:
                scheduler.step()
        self.model_classifier.load_state_dict(best_model_classifier)
        self.model_meta.load_state_dict(best_model_meta)
        final_test = self.test(dataloader_test, n_classes)
        return compute_additional_defer_metrics(final_test)

    def test(self, dataloader, n_classes):
        """
        Test the model
        dataloader: dataloader
        """
        defers_all = []
        truths_all = []
        meta_preds_all = []
        predictions_all = []
        human_preds_all = []
        rej_score1_all = []
        rej_score2_all = []
        class_probs_all = []
        self.model_classifier.eval()
        self.model_meta.eval()

        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)

                # Convert to one-hot

                one_hot_m = torch.zeros((data_x.size()[0], n_classes))
                one_hot_m[torch.arange(data_x.size()[0]), hum_preds] = 1
                one_hot_m = one_hot_m.to(self.device)

                outputs_classifier = F.softmax(self.model_classifier(data_x),
                                               dim=1)
                outputs_meta = F.softmax(self.model_meta(data_x, one_hot_m),
                                         dim=1)
                outputs_defer = F.softmax(self.model_defer(data_x),
                                          dim=1)[:, 1]

                prob_classifier, pred_classifier = \
                    torch.max(outputs_classifier.data, 1)
                _, pred_meta = torch.max(outputs_meta.data, 1)
                prob_posthoc = F.softmax(self.model_defer_meta(data_x),
                                         dim=1)[:, 1]

                _, rejector = torch.max(torch.stack(
                    [prob_classifier, outputs_defer, prob_posthoc]), dim=0)
                rej_sc1 = outputs_defer - prob_classifier
                rej_sc2 = prob_posthoc - prob_classifier

                predictions_all.extend(pred_classifier.cpu().numpy())
                defers_all.extend([int(defer) for defer in
                                   rejector.cpu().numpy()])
                truths_all.extend(data_y.cpu().numpy())
                human_preds_all.extend(hum_preds.cpu().numpy())
                meta_preds_all.extend(pred_meta.cpu().numpy())
                rej_score1_all.extend(rej_sc1.cpu().numpy())
                rej_score2_all.extend(rej_sc2.cpu().numpy())
                class_probs_all.extend(outputs_classifier.cpu().numpy())
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        human_preds_all = np.array(human_preds_all)
        meta_preds_all = np.array(meta_preds_all)
        predictions_all = np.array(predictions_all)
        rej_score1_all = np.array(rej_score1_all)
        rej_score2_all = np.array(rej_score2_all)
        class_probs_all = np.array(class_probs_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "meta_preds": meta_preds_all,
            "preds": predictions_all,
            "rej_score1": rej_score1_all,
            "rej_score2": rej_score2_all,
            "class_probs": class_probs_all,
            "human_preds": human_preds_all,
        }
        return data
