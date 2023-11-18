import copy
import torch
import numpy as np
import torch.nn.functional as F
import logging
from tqdm import tqdm
from Metrics.metrics import compute_deferral_metrics_cost_sensitive
from human_ai_deferral.baselines.compare_confidence import CompareConfidence
from human_ai_deferral.baselines.one_v_all import OVASurrogate
from human_ai_deferral.baselines.lce_surrogate import LceSurrogate
from human_ai_deferral.methods.realizable_surrogate import RealizableSurrogate


class CompareConfidenceCostSensitive(CompareConfidence):
    def __init__(self, model_class, model_expert, device,
                 plotting_interval=100):
        super().__init__(model_class, model_expert, device, plotting_interval)

    def test(self, dataloader, loss_fn):
        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        self.model_expert.eval()
        self.model_class.eval()
        with torch.no_grad():
            for _, (data_x, data_y, hum_preds) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                outputs_expert = self.model_expert(data_x)
                outputs_expert = F.softmax(outputs_expert, dim=1)
                max_class_probs, predicted_class =\
                    torch.max(outputs_class.data, 1)
                class_probs_all.extend(outputs_class.cpu().numpy())
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                defers = []
                for i in range(len(data_y)):
                    rej_score_all.extend(
                        [outputs_expert[i, 1].item() -
                         max_class_probs[i].item()]
                    )
                    if self.is_defer(outputs_expert[i, 1],
                                     outputs_class[i, :],
                                     loss_fn):
                        defers.extend([1])
                    else:
                        defers.extend([0])
                defers_all.extend(defers)
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

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        loss_fn,
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        """fits classifier and expert model

        Args:
            dataloader_train (_type_): train dataloader
            dataloader_val (_type_): val dataloader
            dataloader_test (_type_): _description_
            epochs (_type_): training epochs
            optimizer (_type_): optimizer function
            lr (_type_): learning rate
            costs (_type_): costs array 
            scheduler (_type_, optional): scheduler function. Defaults to None.
            verbose (bool, optional): _description_. Defaults to True.
            test_interval (int, optional): _description_. Defaults to 5.

        Returns:
            dict: metrics on the test set
        """
        optimizer_class = optimizer(self.model_class.parameters(), lr=lr)
        optimizer_expert = optimizer(self.model_expert.parameters(), lr=lr)
        if scheduler is not None:
            scheduler_class = scheduler(optimizer_class,
                                        len(dataloader_train) * epochs)
            scheduler_expert = scheduler(
                optimizer_expert, len(dataloader_train) * epochs
            )
        best_acc = 0
        # store current model dict
        best_model = [copy.deepcopy(self.model_class.state_dict()),
                      copy.deepcopy(self.model_expert.state_dict())]
        for epoch in tqdm(range(epochs)):
            self.fit_epoch_class(
                dataloader_train, optimizer_class, verbose=verbose, epoch=epoch
            )
            self.fit_epoch_expert(
                dataloader_train, optimizer_expert,
                verbose=verbose, epoch=epoch
            )
            if epoch % test_interval == 0 and epoch > 1:
                data_test = self.test(dataloader_val, loss_fn)
                val_metrics = compute_deferral_metrics_cost_sensitive(
                                    data_test, loss_fn)
                if val_metrics["classifier_all_acc"] >= best_acc:
                    best_acc = val_metrics["classifier_all_acc"]
                    best_model = [copy.deepcopy(self.model_class.state_dict()),
                                  copy.deepcopy(self.model_expert.state_dict())
                                  ]

            if scheduler is not None:
                scheduler_class.step()
                scheduler_expert.step()
        self.model_class.load_state_dict(best_model[0])
        self.model_expert.load_state_dict(best_model[1])

        return compute_deferral_metrics_cost_sensitive(
                self.test(dataloader_test, loss_fn), loss_fn)

    def is_defer(self, outputs_expert, outputs_class, loss_matrix):
        '''Decides whether to defer or not based on output of the expert,
            outputs of the classifier, and the costs'''
        # costs = torch.matmul(loss_matrix, )
        # torch.matmul(outputs_class, loss_matrix)
        if outputs_expert > torch.max(outputs_class):
            return True
        else:
            return False


class OVASurrogateCostSensitive(OVASurrogate):
    def __init__(self, alpha, plotting_interval, model,
                 device, learnable_threshold_rej=False):
        super().__init__(alpha, plotting_interval, model,
                         device, learnable_threshold_rej)

    def test(self, dataloader, loss_fn):
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
                outputs_class = F.softmax(outputs[:, :-1], dim=1)
                outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                max_probs, predicted_class = torch.max(outputs.data[:, :-1], 1)
                predictions_all.extend(predicted_class.cpu().numpy())

                # TODO: Rewrite defer_binary
                defer_binary = [self.is_defer(outputs.data[i], loss_fn)
                                for i in range(len(outputs.data))]
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
    
    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        loss_fn,
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
        best_acc = 0
        # store current model dict
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, optimizer, verbose, epoch)
            if epoch % test_interval == 0 and epoch > 1:
                if self.learnable_threshold_rej:
                    self.fit_treshold_rej(dataloader_val)
                data_test = self.test(dataloader_val, loss_fn)
                val_metrics = compute_deferral_metrics_cost_sensitive(
                            data_test, loss_fn)
                if val_metrics["system_acc"] >= best_acc:
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(self.model.state_dict())
                if verbose:
                    logging.info(compute_deferral_metrics_cost_sensitive(
                                data_test, loss_fn))
            if scheduler is not None:
                scheduler.step()
        self.model.load_state_dict(best_model)
        if self.learnable_threshold_rej:
            self.fit_treshold_rej(dataloader_val)
        final_test = self.test(dataloader_test, loss_fn)
        return compute_deferral_metrics_cost_sensitive(final_test, loss_fn)

    def is_defer(self, outputs, loss_fn):
        # TODO: COMPLETE THIS
        pass


class LceSurrogateCostSensitive(LceSurrogate):
    def __init__(self, alpha, plotting_interval, model,
                 device, learnable_threshold_rej=False):
        super().__init__(alpha, plotting_interval, model,
                         device, learnable_threshold_rej)

    def test(self, dataloader, loss_fn):
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
                outputs_class = F.softmax(outputs[:, :-1], dim=1)
                outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                max_probs, predicted_class = torch.max(outputs.data[:, :-1], 1)
                predictions_all.extend(predicted_class.cpu().numpy())
                # TODO: Rewrite defer_binary
                defer_binary = [self.is_defer(outputs.data[i], loss_fn)
                                for i in range(len(outputs.data))]
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
    
    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        loss_fn,
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
        best_acc = 0
        # store current model dict
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, optimizer, verbose, epoch)
            if epoch % test_interval == 0 and epoch > 1:
                if self.learnable_threshold_rej:
                    self.fit_treshold_rej(dataloader_val)
                data_test = self.test(dataloader_val, loss_fn)
                val_metrics = compute_deferral_metrics_cost_sensitive(
                            data_test, loss_fn)
                if val_metrics["system_acc"] >= best_acc:
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(self.model.state_dict())
                if verbose:
                    logging.info(compute_deferral_metrics_cost_sensitive(
                                data_test, loss_fn))
            if scheduler is not None:
                scheduler.step()
        self.model.load_state_dict(best_model)
        if self.learnable_threshold_rej:
            self.fit_treshold_rej(dataloader_val)
        final_test = self.test(dataloader_test, loss_fn)
        return compute_deferral_metrics_cost_sensitive(final_test, loss_fn)

    def fit_hyperparam(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        loss_fn,
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        alpha_grid = [0, 0.5, 1]
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
                loss_fn=loss_fn,
                verbose=verbose,
                test_interval=test_interval,
                scheduler=scheduler,
            )["system_acc"]
            accuracy = compute_deferral_metrics_cost_sensitive(
                        self.test(dataloader_val, loss_fn), loss_fn)[
                            "system_acc"]
            logging.info(f"alpha: {alpha}, accuracy: {accuracy}")
            if accuracy > best_acc:
                best_acc = accuracy
                best_alpha = alpha
        self.alpha = best_alpha
        self.model.load_state_dict(model_dict)
        _ = self.fit(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs,
                optimizer=optimizer,
                lr=lr,
                loss_fn=loss_fn,
                verbose=verbose,
                test_interval=test_interval,
                scheduler=scheduler,
            )
        test_metrics = compute_deferral_metrics_cost_sensitive(self.test(
                dataloader_test, loss_fn), loss_fn)
        return test_metrics
    
    def is_defer(self, outputs, loss_fn):
        # TODO: COMPLETE THIS
        
        pass


class RealizableSurrogateCostSensitive(RealizableSurrogate):
    def test(self, dataloader, loss_fn):
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
                outputs_class = F.softmax(outputs[:, :-1], dim=1)
                outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                max_probs, predicted_class = torch.max(outputs.data[:, :-1], 1)
                predictions_all.extend(predicted_class.cpu().numpy())
                # TODO: Rewrite defer_binary
                defer_binary = [self.is_defer(outputs.data[i], loss_fn)
                                for i in range(len(outputs.data))]
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
    
    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        loss_fn,
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
        best_acc = 0
        # store current model dict
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, optimizer, verbose, epoch)
            if epoch % test_interval == 0 and epoch > 1:
                if self.learnable_threshold_rej:
                    self.fit_treshold_rej(dataloader_val)
                data_test = self.test(dataloader_val, loss_fn)
                val_metrics = compute_deferral_metrics_cost_sensitive(
                            data_test, loss_fn)
                if val_metrics["system_acc"] >= best_acc:
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(self.model.state_dict())
                if verbose:
                    logging.info(compute_deferral_metrics_cost_sensitive(
                            data_test, loss_fn))
            if scheduler is not None:
                scheduler.step()
        self.model.load_state_dict(best_model)
        if self.learnable_threshold_rej:
            self.fit_treshold_rej(dataloader_val)
        final_test = self.test(dataloader_test, loss_fn)
        return compute_deferral_metrics_cost_sensitive(final_test, loss_fn)

    # fit with hyperparameter tuning over alpha
    def fit_hyperparam(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        loss_fn,
        verbose=True,
        test_interval=5,
        scheduler=None,
        alpha_grid=[0, 0.1, 0.3, 0.5, 0.9, 1],
    ):
        # np.linspace(0,1,11)
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
                epochs=epochs,
                optimizer=optimizer,
                lr=lr,
                loss_fn=loss_fn,
                verbose=verbose,
                test_interval=test_interval,
                scheduler=scheduler,
            )["system_acc"]
            accuracy = compute_deferral_metrics_cost_sensitive(
                self.test(dataloader_val, loss_fn), loss_fn)["system_acc"]
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
                epochs=epochs,
                optimizer=optimizer,
                lr=lr,
                loss_fn=loss_fn,
                verbose=verbose,
                test_interval=test_interval,
                scheduler=scheduler,
            )
        test_metrics = compute_deferral_metrics_cost_sensitive(
            self.test(dataloader_test, loss_fn), loss_fn)
        return test_metrics

    def is_defer(self, outputs, loss_fn):
        # TODO: COMPLETE THIS
        
        pass
