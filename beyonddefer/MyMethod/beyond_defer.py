import copy
import torch
import numpy as np
import torch.nn.functional as F
import time
import logging
from tqdm import tqdm
from beyonddefer.human_ai_defer.helpers.utils import AverageMeter, accuracy
from beyonddefer.metrics.metrics import compute_metalearner_metrics
from beyonddefer.human_ai_defer.baselines.basemethod import BaseMethod

eps_cst = 1e-8


class BeyondDefer(BaseMethod):
    def __init__(self, plotting_interval, model_classifier,
                 model_sim, model_meta, device,
                 learnable_threshold_rej=False):
        '''
        plotting_interval (int): used for plotting model training in fit_epoch
        model_classifier (pytorch model): model used for surrogate
        device: cuda device or cpu
        learnable_threshold_rej (bool): whether to learn a treshold on the
        '''
        self.plotting_interval = plotting_interval
        self.model_classifier = model_classifier
        self.model_sim = model_sim
        self.model_meta = model_meta
        self.device = device
        self.threshold_rej = 0
        self.learnable_threshold_rej = learnable_threshold_rej

    def LossOVA(self, outputs, y):
        # outputs[torch.where(outputs == 0.0)] = (-1 * y) * (-1 * np.inf)
        loss_out = torch.log2(1 + torch.exp((-1 * y) * outputs + eps_cst)
                              + eps_cst)
        return loss_out

    def surrogate_loss(self, out_class, outputs_sim, outputs_meta, m,
                       data_y):
        """
        outputs: network outputs
        m: cost of deferring to expert cost of classifier predicting
         hum_preds == target
        labels: target
        """

        batch_size = out_class.size()[0]
        l1 = self.LossOVA(out_class[range(batch_size), data_y], 1)
        l2 = torch.sum(
            self.LossOVA(out_class[range(batch_size), :], -1), dim=1
        ) - self.LossOVA(out_class[range(batch_size), data_y], -1)

        l3 = self.LossOVA(outputs_meta[range(batch_size), data_y], 1)
        l4 = torch.sum(
            self.LossOVA(outputs_meta[range(batch_size), :], -1), dim=1
        ) - self.LossOVA(outputs_meta[range(batch_size), data_y], -1)

        l5 = self.LossOVA(outputs_sim[range(batch_size), m], 1)
        l6 = torch.sum(
            self.LossOVA(outputs_sim[range(batch_size), :], -1), dim=1
        ) - self.LossOVA(outputs_sim[range(batch_size), m], -1)

        loss_final = l1 + l2 + l3 + l4 + l5 + l6
        if torch.isnan(loss_final).any():
            ls = [l1, l2, l3, l4, l5, l6]
            for i, l in enumerate(ls):
                if torch.isnan(l).any():
                    print("loss l", i, " has nan")
        return torch.mean(loss_final)

    def LossBCE(self, outputs, y):
        num_classes = outputs.size()[1]
        y_oh = F.one_hot(y, num_classes=num_classes).float()
        return F.binary_cross_entropy_with_logits(outputs, y_oh,
                                                  reduction='none').sum(dim=1)

    def surrogate_loss_bce(self, out_class, outputs_sim, outputs_meta, m,
                           data_y):
        """
        outputs: network outputs
        m: cost of deferring to expert cost of classifier predicting
         hum_preds == target
        labels: target
        """
        return (self.LossBCE(out_class, data_y)
                + self.LossBCE(outputs_sim, m)
                + self.LossBCE(outputs_meta, data_y)).mean()

    def fit_epoch(self, dataloader, n_classes, optimizer, verbose=False,
                  epoch=1):
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
        top1_sim = AverageMeter()
        top1_meta = AverageMeter()
        end = time.time()
        self.model_classifier.train()
        self.model_sim.train()
        self.model_meta.train()

        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)

            # TODO: Check if hum_preds is one-hot or not (probably not)
            # print("shape of hum_preds: ", hum_preds.shape)
            # Convert to one-hot

            one_hot_m = torch.zeros((data_x.size()[0], n_classes))
            one_hot_m[torch.arange(data_x.size()[0]), hum_preds] = 1
            one_hot_m = one_hot_m.to(self.device)

            outputs_classifier = self.model_classifier(data_x)
            outputs_meta = self.model_meta(data_x, one_hot_m)
            outputs_sim = self.model_sim(data_x)

            loss = self.surrogate_loss_bce(outputs_classifier, outputs_sim,
                                           outputs_meta, hum_preds, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1_classifier = accuracy(outputs_classifier.data, data_y,
                                        topk=(1,))[0]
            prec1_meta = accuracy(outputs_meta.data, data_y, topk=(1,))[0]
            prec1_sim = accuracy(outputs_sim.data, hum_preds, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1_classifier.update(prec1_classifier.item(), data_x.size(0))
            top1_sim.update(prec1_sim.item(), data_x.size(0))
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
                    "Prec@1 Sim {top1_sim.val:.3f} ({top1_sim.avg:.3f})\t"
                    "Prec@1 Meta {top1_meta.val:.3f} \
                                    ({top1_meta.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1_classifier=top1_classifier,
                        top1_sim=top1_sim,
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
                        list(self.model_sim.parameters()) + \
                        list(self.model_meta.parameters())
        optimizer = optimizer(params, lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
        best_acc = 0

        # store current model dict
        best_model_classifier = copy.deepcopy(
                                self.model_classifier.state_dict())
        best_model_sim = copy.deepcopy(self.model_sim.state_dict())
        best_model_meta = copy.deepcopy(self.model_meta.state_dict())
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, n_classes, optimizer, verbose,
                           epoch)
            if epoch % test_interval == 0 and epoch > 1:
                data_test = self.test(dataloader_val, n_classes)
                val_metrics = compute_metalearner_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:
                    logging.info("New best model")
                    best_acc = val_metrics["system_acc"]
                    best_model_classifier = copy.deepcopy(
                                    self.model_classifier.state_dict())
                    best_model_sim = copy.deepcopy(self.model_sim.state_dict())
                    best_model_meta = copy.deepcopy(
                                    self.model_meta.state_dict())
                if verbose:
                    logging.info(compute_metalearner_metrics(data_test))
            if scheduler is not None:
                scheduler.step()
        self.model_classifier.load_state_dict(best_model_classifier)
        self.model_sim.load_state_dict(best_model_sim)
        self.model_meta.load_state_dict(best_model_meta)
        if self.learnable_threshold_rej:
            self.fit_treshold_rej(dataloader_val)
        final_test = self.test(dataloader_test, n_classes)
        return compute_metalearner_metrics(final_test)

    def test(self, dataloader, n_classes):
        """
        Test the model
        dataloader: dataloader
        """
        defers_all = []
        truths_all = []
        meta_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model_classifier.eval()
        self.model_meta.eval()
        self.model_sim.eval()

        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)

                # Convert to one-hot

                one_hot_m = torch.zeros((data_x.size()[0], n_classes))
                one_hot_m[torch.arange(data_x.size()[0]), hum_preds] = 1
                one_hot_m = one_hot_m.to(self.device)

                outputs_classifier = F.sigmoid(self.model_classifier(data_x))
                # normalize
                outputs_classifier /= torch.sum(outputs_classifier, axis=1).\
                    unsqueeze(1)
                outputs_meta = F.sigmoid(self.model_meta(data_x, one_hot_m))
                outputs_meta /= torch.sum(outputs_meta, axis=1).unsqueeze(1)
                outputs_sim = F.sigmoid(self.model_sim(data_x))
                outputs_sim /= torch.sum(outputs_sim, axis=1).unsqueeze(1)

                prob_classifier, pred_classifier = \
                    torch.max(outputs_classifier.data, 1)
                _, pred_meta = torch.max(outputs_meta.data, 1)
                prob_posthoc = torch.zeros((hum_preds.size(0))).to(self.device)

                for j in range(n_classes):
                    one_hot_j = torch.zeros((hum_preds.size(0), n_classes))
                    one_hot_j[:, j] = 1
                    one_hot_j = one_hot_j.to(self.device)

                    outputs_meta_j = F.sigmoid(self.model_meta(data_x,
                                                               one_hot_j))
                    prob_meta_j, _ = torch.max(outputs_meta_j.data, 1)
                    prob_posthoc += outputs_sim[:, j] * prob_meta_j

#               rejector = prob_posthoc - cost - prob_ai

                rejector = prob_posthoc - prob_classifier
#               predictions = pred_classifier * (rejector <= 0) + pred_meta *
#               (rejector > 0)

                predictions_all.extend(pred_classifier.cpu().numpy())
                defers_all.extend([int(defer) for defer in (rejector > 0)])
                truths_all.extend(data_y.cpu().numpy())
                meta_preds_all.extend(pred_meta.cpu().numpy())
                rej_score_all.extend(rejector.cpu().numpy())
                class_probs_all.extend(outputs_classifier.cpu().numpy())


#         Note: in our case, hum_preds is actually the meta_pred because we
#         are not deferring to human and we defer to meta learner instead
#         convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        meta_preds_all = np.array(meta_preds_all)
        predictions_all = np.array(predictions_all)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "meta_preds": meta_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
        }
        return data
