import haiid
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import RocCurveDisplay
# from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

class data_prep():

    def __init__(self, name):
        self.task_name = name
        data = self.set_task_data()
        self.train, self.test = self.test_train(data, 0.2)


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
        
        # self.discretize_human_estimates()

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

    # divide the data into train and test set randomly
    def test_train(self, data, prop):

        df = data
        
        # find the size of df
        size_all = df['task_instance_id'].nunique()

        # find the number of test data
        size_test = int(size_all * prop)

        # find the number of train data
        size_train = size_all - size_test

        # In the following, I assume that there is no multiple annotated data

        # set test data
        test = df.sample(n=size_test, random_state=1)

        # set train data
        train = df.drop(test.index)

        return (train, test)

