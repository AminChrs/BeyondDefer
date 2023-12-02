import sys
sys.path.insert(0, './human_ai_defer/')
# from Experiments.SampleComp import SampleComp_par
# from Experiments.CIFAR10K import Exp_parallel
# from Experiments.no_loss_acc_cov import cov_acc_parallel
# from Experiments.acc_vs_c import acc_c_parallel
from tests.test import test_all
test_all()
# iter = int(sys.argv[1])
# Exp_parallel(array[iter])
# cov_acc_parallel(array[iter])
# acc_c_parallel(iter)
# SampleComp_par(array[iter])
