# In this file, I run the calib_prep file and evaluate the data

import calib_prep
import numpy as np

# load all data
data = calib_prep.data_prep('census')
train_data = data.train
test_data = data.test

# get the number of unique task instances within train data
train_task_ids = train_data['task_instance_id'].unique()
num_train_task = len(train_task_ids)
print("The number of unique task instances within train data is: ", num_train_task)
print("tasks: ", train_data['task_instance_id'])
print("participants: ", train_data['participant_id'])
# get the number of annotators
num_annotators = len(train_data['participant_id'].unique())
print("The number of annotators is: ", num_annotators)

# get the number of unique task instances within test data
test_task_ids = test_data['task_instance_id'].unique()
num_test_task = len(test_task_ids)

# get the number of annotators within test data
test_annotators = test_data['participant_id'].unique()

# Figure out whether or not for the same task instance, there are multiple annotators
num_annotators = np.zeros([num_train_task, 1])
# print("train_data['task_instance_id']: ", train_data['task_instance_id'])
# find all keys in train_data['task_instance_id']  
keys = train_data['task_instance_id'].keys()
for i in range(len(keys)):
    # instance_id = train_data['task_instance_id'][keys[i]]
    # print("instance_id: ", instance_id)
    print("keys[i]: ", keys[i])
    num_annotators[keys[i]] += 1

if (num_annotators == 1).all():
    print("There is no multiple annotators for the same task instance")
else:
    # find the instance with more than one annotator
    instance_id = np.where(num_annotators > 1)
    print("There are "+str(num_annotators[instance_id])+" annotators for the same task instance"+str(instance_id))




