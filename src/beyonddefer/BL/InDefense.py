import copy
import torch
import numpy as np
import torch.nn.functional as F
import time
import logging
from tqdm import tqdm
from beyonddefer.human_ai_defer.helpers.utils import AverageMeter, accuracy
from beyonddefer.metrics.metrics import compute_additional_defer_metrics
from beyonddefer.human_ai_defer.baselines.basemethod import BaseMethod

eps_cst = 1e-8


class InDefense(BaseMethod):
    def __init__(self, plotting_interval, model, device):
        '''
        plotting_interval (int): used for plotting model training in fit_epoch
        model_classifier (pytorch model): model used for surrogate
        device: cuda device or cpu
        '''
        self.plotting_interval = plotting_interval
        self.model = model
        self.device = device
        self.threshold_rej = 0

    def asymmetric_softmax(self, model_output, n_classes):
        return_tensor = torch.zeros_like(model_output)
        assert len(model_output.shape) == 2
        return_tensor[:, :n_classes+1] = \
            F.softmax(model_output[:, :n_classes+1], dim=1)
        max_model_output, _ = \
            torch.max(torch.exp(model_output[:, :n_classes+1]), dim=1,
                      keepdim=True)
        sum_model_outputs = \
            torch.sum(torch.exp(model_output), dim=1, keepdim=True)
        return_tensor[:, -1] = \
            torch.exp(model_output[:, -1])/(sum_model_outputs -
                                            max_model_output)
        return return_tensor

    def LossBCEH(self, outputs, y, m):
        num_classes = outputs.size()[1]
        output_human = self.asymmetric_softmax(outputs, num_classes)
        output_human = output_human[:, -1]
        y_oh = (m == y).float()
        return F.binary_cross_entropy(outputs, y_oh,
                                      reduction='none').sum(dim=1)

    def LossBCE(self, outputs, y):
        num_classes = outputs.size()[1]
        y_oh = F.one_hot(y, num_classes=num_classes).float()
        return F.binary_cross_entropy_with_logits(outputs, y_oh,
                                                  reduction='none').sum(dim=1)

    def surrogate_loss_bce(self, out_class,  m, data_y):
        """
        outputs: network outputs
        m: cost of deferring to expert cost of classifier predicting
         hum_preds == target
        labels: target
        """
        return (self.LossBCEH(out_class, data_y, m)
                + self.LossBCE(out_class[:, :-1], data_y)).mean()

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
            one_hot_m = torch.zeros((data_x.size()[0], n_classes))
            one_hot_m[torch.arange(data_x.size()[0]), hum_preds] = 1
            one_hot_m = one_hot_m.to(self.device)

            outputs = self.model(data_x)

            loss = self.surrogate_loss_bce(outputs, hum_preds, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1_classifier = accuracy(outputs[:, :-1].data,
                                        data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1_classifier.update(prec1_classifier.item(), data_x.size(0))

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
        params = list(self.model.parameters())
        optimizer = optimizer(params, lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
        best_acc = 0

        # store current model dict
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, n_classes, optimizer, verbose,
                           epoch)
            if epoch % test_interval == 0 and epoch > 1:
                data_test = self.test(dataloader_val, n_classes)
                val_metrics = compute_additional_defer_metrics(data_test)
                if val_metrics["system_acc"] >= best_acc:
                    logging.info("New best model")
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(
                                    self.model.state_dict())
                if verbose:
                    logging.info(compute_additional_defer_metrics(data_test))
                    data_test = self.test(dataloader_test, n_classes)
                    logging.info(compute_additional_defer_metrics(data_test))
            if scheduler is not None:
                scheduler.step()
        self.model.load_state_dict(best_model)
        final_test = self.test(dataloader_test, n_classes)
        return compute_additional_defer_metrics(final_test)

    def test(self, dataloader, n_classes):
        """
        Test the model
        dataloader: dataloader
        """
        defers_all = []
        truths_all = []
        human_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model.eval()

        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)

                # Convert to one-hot

                one_hot_m = torch.zeros((data_x.size()[0], n_classes))
                one_hot_m[torch.arange(data_x.size()[0]), hum_preds] = 1
                one_hot_m = one_hot_m.to(self.device)

                outputs_classifier = self.asymmetric_softmax(
                            self.model(data_x))
                outputs_classifier = outputs_classifier[:, :-1]
                output_defer = outputs_classifier[:, -1]

                prob_classifier, pred_classifier = \
                    torch.max(outputs_classifier.data, 1)

                _, rejector = torch.max(torch.stack(
                    [prob_classifier, output_defer]), dim=0)
                # rejector is 0 if classifier, 1 if defer, 2 if posthoc
                rej_sc = output_defer - prob_classifier

                predictions_all.extend(pred_classifier.cpu().numpy())
                defers_all.extend([int(defer) for defer in
                                   rejector.cpu().numpy()])
                truths_all.extend(data_y.cpu().numpy())
                human_preds_all.extend(hum_preds.cpu().numpy())
                rej_score_all.extend(rej_sc.cpu().numpy())
                class_probs_all.extend(outputs_classifier.cpu().numpy())


#         Note: in our case, hum_preds is actually the meta_pred because we
#         are not deferring to human and we defer to meta learner instead
#         convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        predictions_all = np.array(predictions_all)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        human_preds_all = np.array(human_preds_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
            "human_preds": human_preds_all
        }
        return data
