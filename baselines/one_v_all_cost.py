import torch
import numpy as np
import torch.nn.functional as F
from human_ai_defer.baselines.basemethod import BaseSurrogateMethod

eps_cst = 1e-8


class OVACost(BaseSurrogateMethod):
    """Method of OvA surrogate from Calibrated Learning to Defer with
    One-vs-All Classifiers
    https://proceedings.mlr.press/v162/verma22c/verma22c.pdf"""
    # from https://github.com/rajevv/OvA-L2D/blob/main/losses/losses.py
    def LogisticLossOVA(self, outputs, y):
        outputs[torch.where(outputs == 0.0)] = (-1 * y) * (-1 * np.inf)
        loss = torch.log2(1 + torch.exp((-1 * y) * outputs + eps_cst) +
                          eps_cst)
        return loss

    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        """
        outputs: network outputs
        m: cost of deferring to expert cost of classifier predicting
            hum_preds == target
        labels: target
        """
        human_conf = (1-self.loss_cost_sensitive(self.loss_matrix,
                      data_y, hum_preds))
        batch_size = outputs.size()[0]
        l1 = self.LogisticLossOVA(outputs[range(batch_size), data_y], 1)
        l2 = torch.sum(
            self.LogisticLossOVA(outputs[:, :-1], -1), dim=1
        ) - self.LogisticLossOVA(outputs[range(batch_size), data_y], -1)
        l3 = self.LogisticLossOVA(outputs[range(batch_size), -1], -1)
        l4 = self.LogisticLossOVA(outputs[range(batch_size), -1], 1)

        l5 = human_conf * (l4 - l3)

        loss = l1 + l2 + l3 + l5

        return torch.mean(loss)

    def test(self, dataloader):
        """
        Test the model
        dataloader: dataloader
        """
        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)

                outputs = self.model(data_x)
                outputs_class = F.sigmoid(outputs[:, :-1])
                outputs = F.sigmoid(outputs)
                max_probs, predicted_class = \
                    torch.max(1 - torch.matmul(outputs_class.data,
                                               self.loss_matrix),
                              1)
                predictions_all.extend(predicted_class.cpu().numpy())
                defer_scores = [outputs.data[i][-1].item() -
                                outputs.data[i][predicted_class[i]].item() for
                                i in range(len(outputs.data))]
                defer_binary = [int(defer_score >= self.threshold_rej) for
                                defer_score in defer_scores]
                defers_all.extend(defer_binary)
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                for i in range(len(outputs.data)):
                    rej_score_all.append(
                        outputs.data[i][-1].item()
                        - outputs.data[i][predicted_class[i]].item()
                    )
                class_probs_all.extend(outputs_class.cpu().numpy())

        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
        }
        return data

    def set_loss_matrix(self, loss_matrix):
        self.loss_matrix = loss_matrix.to(self.device)

    def loss_cost_sensitive(self, loss_matrix, y, y_pred):
        num_classes = loss_matrix.shape[0]
        y_oh = F.one_hot(y, num_classes=num_classes).float()
        yp_oh = F.one_hot(y_pred, num_classes=num_classes).float()
        is_loss = torch.matmul(torch.matmul(y_oh, loss_matrix),
                               yp_oh.transpose(0, 1))
        is_loss = torch.diag(is_loss).float()
        return is_loss
