import copy
import math
from pyexpat import model
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
import shutil
import time
import torch.utils.data as data
import sys
import pickle
import logging
from tqdm import tqdm

sys.path.append("..")
from human_ai_deferral.helpers.utils import *
from human_ai_deferral.helpers.metrics import *
from human_ai_deferral.baselines.basemethod import BaseMethod, BaseSurrogateMethod

eps_cst = 1e-8


        
class CostyDeferral:
    def __init__(self, original_class, *args):        
        # Compare Confidence Modification
        def modify_compare_confidence(self):
            '''The method, modifies "test" and "fit" methods for compare confidence
            to support the additional cost of defer "c" '''
            # modified test
            def test(self, dataloader, c=0):
                defers_all = []
                truths_all = []
                hum_preds_all = []
                predictions_all = []  # classifier only
                rej_score_all = []  # rejector probability
                class_probs_all = []  # classifier probability
                self.model_expert.eval()
                self.model_class.eval()
                with torch.no_grad():
                    for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
                        data_x = data_x.to(self.device)
                        data_y = data_y.to(self.device)
                        hum_preds = hum_preds.to(self.device)
                        outputs_class = self.model_class(data_x)
                        outputs_class = F.softmax(outputs_class, dim=1)
                        outputs_expert = self.model_expert(data_x)
                        outputs_expert = F.softmax(outputs_expert, dim=1)
                        max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                        class_probs_all.extend(outputs_class.cpu().numpy())
                        predictions_all.extend(predicted_class.cpu().numpy())
                        truths_all.extend(data_y.cpu().numpy())
                        hum_preds_all.extend(hum_preds.cpu().numpy())
                        defers = []
                        for i in range(len(data_y)):
                            # add c in rej score
                            rej_score_all.extend(
                                [outputs_expert[i, 1].item() - max_class_probs[i].item() - c]
                            )
                            # add c in decision to defer here
                            if outputs_expert[i, 1] > (max_class_probs[i] + c):
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
            
            
            # modified fit
            def fit(
                self,
                dataloader_train,
                dataloader_val,
                dataloader_test,
                epochs,
                optimizer,
                lr,
                c = 0,
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
                    c (_type_): cost of defer
                    scheduler (_type_, optional): scheduler function. Defaults to None.
                    verbose (bool, optional): _description_. Defaults to True.
                    test_interval (int, optional): _description_. Defaults to 5.

                Returns:
                    dict: metrics on the test set
                """
                optimizer_class = optimizer(self.model_class.parameters(), lr=lr)
                optimizer_expert = optimizer(self.model_expert.parameters(), lr=lr)
                if scheduler is not None:
                    scheduler_class = scheduler(optimizer_class, len(dataloader_train) * epochs)
                    scheduler_expert = scheduler(
                        optimizer_expert, len(dataloader_train) * epochs
                    )
                best_acc = 0
                # store current model dict
                best_model = [copy.deepcopy(self.model_class.state_dict()), copy.deepcopy(self.model_expert.state_dict())]
                for epoch in tqdm(range(epochs)):
                    self.fit_epoch_class(
                        dataloader_train, optimizer_class, verbose=verbose, epoch=epoch
                    )
                    self.fit_epoch_expert(
                        dataloader_train, optimizer_expert, verbose=verbose, epoch=epoch
                    )
                    if epoch % test_interval == 0 and epoch > 1:
                        data_test = self.test(dataloader_val)
                        val_metrics = compute_deferral_metrics(data_test)
                        if val_metrics["classifier_all_acc"] >= best_acc: 
                            best_acc = val_metrics["classifier_all_acc"]
                            best_model = [copy.deepcopy(self.model_class.state_dict()), copy.deepcopy(self.model_expert.state_dict())]

                    if scheduler is not None:
                        scheduler_class.step()
                        scheduler_expert.step()
                self.model_class.load_state_dict(best_model[0])
                self.model_expert.load_state_dict(best_model[1])

                return compute_deferral_metrics(self.test(dataloader_test, c))

            
            self.model_class = args[0]
            self.model_expert = args[1]
            self.device = args[2]
            if len(args) == 4:
                self.plotting_interval = args[3]
            else:
                self.plotting_interval = 100
            
            setattr(self, 'test', test)
            setattr(self, 'fit', fit)
            return 

        # OvA Modification
        def modify_ova(self):
            pass
        
        # RS Modification
        def modify_rs(self):
            pass
        
        # LCE Modification
        def modify_lce(self):
            pass
        
        
          
        self.original_class = original_class
        if original_class.__name__ == "CompareConfidence":
            modify_compare_confidence(self)
        elif original_class.__name__ == "OVASurrogate":
            modify_ova(self)
        elif original_class.__name__ == "RealizableSurrogate":
            modify_rs(self)
        elif original_class.__name__ == "LceSurrogate":
            modify_lce(self)

        
            
    def __getattr__(self, attr):
        # Delegate all other attribute access to the original class
        return getattr(self.original_class, attr)
        

            