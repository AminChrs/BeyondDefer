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
from .basemethod import BaseMethod, BaseSurrogateMethod

eps_cst = 1e-8


class CostyDeferral:
    def __init__(self, base_class):
        self.base_class = base_class

        # Dynamically add attributes and methods from the base class to self
        for attr_name in dir(base_class):
            attr = getattr(base_class, attr_name)
            if callable(attr) or isinstance(attr, property):
                setattr(self, attr_name, attr)
        
        # ? What methods should be changed for each class?
        # - compare_confidence: self.test()
        # - beyond_defer: 
        # - differentiable_triage: 
        # - lce_surrogate:
        # - mix_of_exps:
        # - one_v_all:
        
        
        if base_class.__name__ == "CompareConfidence":
            def new_test(self, dataloader, cost):
                # the new test method should be written here
                pass
        
            # Replace the original method with the new one
            setattr(self, "test", new_test)
        
        elif base_class.__name__ == "BeyondDefer":
            pass
        
        elif base_class.__name__ == "LceSurrogate":
            pass