import haiid
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import RocCurveDisplay
# from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

class data_prep:

    def _init_(self, name):
        self.task_name = name


    def set_task_data(self):

        # load all data
        df = haiid.load_dataset("./")
        # get specified task data
        task_df = haiid.load_task(df, self.task_name)
        if self.task_name == 'census':
            self.task_data = self.preprocess_data(task_df, '>=50k')
        elif self.task_name == 'sarcasm':
            self.task_data = self.preprocess_data(task_df, 'sarcasm')
        else:
            self.task_data  = self.preprocess_data(task_df)
        
        self.discretize_human_estimates()

        return (self.task_data)

    def preprocess_data(self, data, label_1=None):

        #filter data for 'geographic region' and 'perceived accuracy'
        df = data.loc[(data['geographic_region']=='United States') & (data['perceived_accuracy']==80)]
        #select relevant data columns
        df = df[['task_instance_id','participant_id','correct_label', 'advice', 'response_1','response_2']] 

        #assign event Y=1
        if label_1 == None:
            #assigning event Y=1 to a random label per task instance
            #randomness needed since there are 4 labels accross tasks, but each task has two labels -> assign one of them to event Y=1 
            np.random.seed(320)
            task_ids = df['task_instance_id'].unique()
            df_label = pd.DataFrame(task_ids, columns=['task_instance_id'])
            df_label['y'] = np.random.choice(2, len(task_ids)).astype(int) 
            df = df.merge(df_label, how='left', on='task_instance_id' )
        else:
            #assigning event Y=1 to label specified by label_1 
            df['y'] = (df['correct_label']==label_1).astype(int)

        #compute mapping from [-1,1] to [0,1]
        df[['b','h','h+AI']] = (df.loc[:,[ 'advice','response_1', 'response_2']]+1)/ 2.0 
        df.loc[df['y']==0,['b', 'h', 'h+AI']] = 1 - df.loc[df['y']==0,['b','h','h+AI']]

        return(df[['task_instance_id','participant_id','h','b','y','h+AI']])

