import sys
import beyonddefer
from beyonddefer.Tests.test import test_all
# from beyonddefer.Experiments.SampleComp import SampleComp_par
# from beyonddefer.Experiments.CIFAR10K import Exp_parallel
# from beyonddefer.Experiments.no_loss_acc_cov import cov_acc_parallel
# from beyonddefer.Experiments.acc_vs_c import acc_c_parallel


def main():
    """Main Function"""
    test_all()


if __name__ == '__main__':
    main()
# iter = int(sys.argv[1])
# Exp_parallel(array[iter])
# cov_acc_parallel(array[iter])
# acc_c_parallel(iter)
# SampleComp_par(array[iter])
