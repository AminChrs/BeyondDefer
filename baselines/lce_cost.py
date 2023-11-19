import copy
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from human_ai_defer.helpers.metrics import compute_deferral_metrics
from human_ai_defer.baselines.basemethod import BaseSurrogateMethod


eps_cst = 1e-8


class LceCost(BaseSurrogateMethod):
    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        """
        Implmentation of L_{CE}^{\alpha}
        """
        outputs = F.softmax(outputs, dim=1)
        y_oh = F.one_hot(data_y, num_classes=outputs.size()[1] - 1).float()
        m_oh = F.one_hot(hum_preds, num_classes=outputs.size()[1] - 1).float()
        cost_human = torch.matmul(torch.matmul(y_oh, self.loss_matrix),
                                  m_oh.transpose(0, 1))
        cost_human = torch.diag(cost_human)
        cost_y = torch.matmul(y_oh, self.loss_matrix)
        max_cost_y, _ = torch.max(cost_y, dim=1)
        max_cost = torch.max(cost_human, max_cost_y)
        human_correct = max_cost - cost_human
        y_correct = max_cost.unsqueeze(1).repeat(1, outputs.size()[1] - 1)\
            - cost_y
        m2 = self.alpha * human_correct + (1 - human_correct)
        human_correct = torch.tensor(human_correct).to(self.device)
        m2 = torch.tensor(m2).to(self.device)
        batch_size = outputs.size()[0]  # batch_size
        loss_1 = -human_correct * torch.log2(
            outputs[range(batch_size), -1] + eps_cst
        )
        loss_2 = -m2 * torch.sum(y_correct * torch.log2(
            outputs[range(batch_size), :-1] + eps_cst
        ), axis=1)  # pick the values corresponding to the labels
        return torch.sum(loss_1 + loss_2) / batch_size

    # fit with hyperparameter tuning over alpha
    def fit_hyperparam(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        alpha_grid = [0, 0.5, 1]  # 0, 0.5, 1
        best_alpha = 0
        best_acc = 0
        model_dict = copy.deepcopy(self.model.state_dict())
        for alpha in tqdm(alpha_grid):
            self.alpha = alpha
            self.model.load_state_dict(model_dict)
            self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs,
                optimizer=optimizer,
                lr=lr,
                verbose=verbose,
                test_interval=test_interval,
                scheduler=scheduler,
            )["system_acc"]
            accuracy = compute_deferral_metrics(self.test(
                            dataloader_val))["system_acc"]
            logging.info(f"alpha: {alpha}, accuracy: {accuracy}")
            if accuracy > best_acc:
                best_acc = accuracy
                best_alpha = alpha
        self.alpha = best_alpha
        self.model.load_state_dict(model_dict)
        self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs,
                optimizer=optimizer,
                lr=lr,
                verbose=verbose,
                test_interval=test_interval,
                scheduler=scheduler,
            )
        test_metrics = compute_deferral_metrics(self.test(dataloader_test))
        return test_metrics

    def set_loss_matrix(self, loss_matrix):
        self.loss_matrix = loss_matrix.to(self.device)
