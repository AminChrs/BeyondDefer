#from Experiments.SampleComp import *
from Tests.test import test_CC_cost_sensitive_deferral
import numpy as np
from Metrics.metrics import compute_deferral_metrics_cost_sensitive
test_CC_cost_sensitive_deferral()
#compute_deferral_metrics_cost_sensitive(1,2)
# arr1 = np.array([1,2,3,4])
# arr2 = np.array([5,6,7,8])
# arr3 = np.concatenate([np.reshape(arr1, (arr1.shape[0], 1)), np.reshape(arr2, (arr2.shape[0],1))], axis=1)

# print(np.sum(np.apply_along_axis(lambda x: x[0] + x[1], 1, arr3)))
