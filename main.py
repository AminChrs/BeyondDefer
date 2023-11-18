# from Experiments.SampleComp import SampleComp_par
# from Experiments.CIFAR10K import Exp_parallel
# from Experiments.no_loss_acc_cov import cov_acc_parallel
from Experiments.acc_vs_c import acc_c_parallel
# from Tests.test import *
import sys
iter = int(sys.argv[1])
# array = [1, 3, 4, 5, 6, 7, 8, 9, 10]
# array = [149, 66, 72]
# Exp_parallel(array[iter])
# cov_acc_parallel(array[iter])
acc_c_parallel(iter)
# SampleComp_par(array[iter])
