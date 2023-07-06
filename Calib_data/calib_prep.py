import haiid
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_task_data(self):

    # load all data
    df = haiid.load_dataset("./human_ai_interactions_data")
    # get specified task data
    task_df = haiid.load_task(df, self.task_name)
    if self.task_name =='census':
        self.task_data = self.preprocess_data(task_df, '>=50k')
    elif self.task_name =='sarcasm':
        self.task_data = self.preprocess_data(task_df, 'sarcasm')
    else:
        self.task_data  = self.preprocess_data(task_df)
    
    self.discretize_human_estimates()

    return (self.task_data)