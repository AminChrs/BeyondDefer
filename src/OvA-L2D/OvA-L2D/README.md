> Rajeev Verma, Eric Nalisnick. "Calibrated Learning to Defer with One-vs-All Classifiers." https://arxiv.org/abs/2202.03673

In this paper, we propose an alternate One-vs-All loss function parameterization for the Learning to Defer (L2D) problem. The proposed loss function is a consistent surrogate loss function for 0-1 misclassification L2D loss with improved calibration of confidence estimates with respect to the expert correctness. 

## Setup
First set up a conda environment. We provide the environment yml file: `defer.yml`. Next, create a directory `./Data` and keep datasets in it. 

## Starter Guide
The implementation of the loss function(s) is available in `./losses`. The main script to execute training is `./main.py`. We provide the usage guide below. Note specifically the flag `--loss_type` which one can set to `softmax` or `ova`. Trained models will be saved in a (sub)directory with the name `loss_type` in the `ckp_dir` directory. 

```
usage: main.py [-h] [--batch_size BATCH_SIZE] [--alpha ALPHA] [--epochs EPOCHS] [--patience PATIENCE] [--expert_type EXPERT_TYPE]
               [--n_classes N_CLASSES] [--k K] [--lr LR] [--weight_decay WEIGHT_DECAY] [--warmup_epochs WARMUP_EPOCHS]
               [--loss_type LOSS_TYPE] [--ckp_dir CKP_DIR] [--experiment_name EXPERIMENT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --alpha ALPHA         scaling parameter for the loss function, default=1.0.
  --epochs EPOCHS
  --patience PATIENCE   number of patience steps for early stopping the training.
  --expert_type EXPERT_TYPE
                        specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.
  --n_classes N_CLASSES
                        K for K class classification.
  --k K
  --lr LR               learning rate.
  --weight_decay WEIGHT_DECAY
  --warmup_epochs WARMUP_EPOCHS
  --loss_type LOSS_TYPE
                        surrogate loss type for learning to defer.
  --ckp_dir CKP_DIR     directory name to save the checkpoints.
  --experiment_name EXPERIMENT_NAME
                        specify the experiment name. Checkpoints will be saved with this name.
``` 

## Citation
```
@inproceedings{Verma2022Calibrated,
  title = {Calibrated Learning to Defer with One-vs-All Classifiers},
  author = {Verma, Rajeev and Nalisnick, Eric},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning (ICML)},
  year = {2022}
}
```


## Acknowledgements 
As with everything else, the code in this repo is built upon the excellent works of other researchers. We greatly acknowledge Hussein Mozannar and David Sontag. Their code for the paper Consistent Estimators for Learning to Defer (https://github.com/clinicalml/learn-to-defer) formed the basis of this repository. Additionally, we use code from Nastaran Okati et al. (https://github.com/Networks-Learning/differentiable-learning-under-triage) and Matthijs Hollemans (https://github.com/hollance/reliability-diagrams). 
