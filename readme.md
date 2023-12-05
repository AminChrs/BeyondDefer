# Quick Start

In this package, we provide the code to reproduce the experiments in the paper "Is Learn to Defer Enough? Optimal Predictors that Incorporate Human Decisions". 

# Quick Installation
## Using pip (Recommended)
This package could be installed easily using pip with the following commands. It is recommended that you create a virtual environment and install everything there:
```sh
# Creating a virtual environment (optional)
python3 -m venv beyonddefer-venv
source beyonddefer-venv/bin/activate

# Adding the Package
pip install beyonddefer
```

## Cloning Repository
Another way to use the package is to clone this repository and then add the package's path to the python path (using `PYTHONPATH` environmental variable):

```sh
# cloning the repositiry and installing requirements
git clone <repo-url>
cd BeyondDefer
pip install -r requirements.txt

# adding the package to python path
export PYTHONPATH=$PWD

# run your python script which includes beyonddefer
```

# Usage Example
In this section, we go through an example of using `beyonddefer` package and writing a simple python code. In this example, we simply train the `Additional Beyond Defer` method with `WideResNet` model with the synthetic `CIFAR10K` dataset for `k = 5`. Then we test for the results and print them out.

**Note:** Before running any experiments, you should first create the `data`, `models`, and `Results` directories in the directory of your python script:
```sh
mkdir data models Results
```

Here is code for the introduced example:
```python
# import beyonddefer itself for some initializations
import beyonddefer

# import the required modules
from beyonddefer.human_ai_defer.datasetsdefer.cifar_synth import \
    CifarSynthDataset
from beyonddefer.MyMethod.additional_defer import AdditionalBeyond
from beyonddefer.MyNet.call_net import networks, optimizer_scheduler
from beyonddefer.metrics.metrics import compute_additional_defer_metrics
import torch
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting log level to INFO to see training logs
logging.getLogger().setLevel(logging.INFO)

# Adding the dataset
k = 5 # expert k
dataset = CifarSynthDataset(k, False, batch_size=512)

dataset_name = "cifar_synth"
epochs = 150
num_classes = 10

# Adding the networks and model
classifier, human, meta = networks(dataset_name, "Additional", device)
AB = AdditionalBeyond(10, classifier, human, meta, device)

# Optimizer and scheduler
optimizer, scheduler = optimizer_scheduler()

# Training the model
AB.fit(dataset.data_train_loader, dataset.data_val_loader,
           dataset.data_test_loader, num_classes, epochs, optimizer, lr=1e-3,
           scheduler=scheduler, verbose=False)

# Generating test results
test_data = AB.test(dataset.data_test_loader, num_classes)

# Extracting useful information from the raw test data
res_AB = compute_additional_defer_metrics(test_data)

print(res_AB)
```

## Experiments
The main set of experiments shown in the paper are in `Experiments/` (Section 7). In fact,

- in `Experiments/acc_vs_c.py`
the code corresponding to the accuracy of methods based on additional defer cost is provided,
- in `Experiments/CIFAR10K.py`
the code corresponding to the CIFAR10K experiment for different $K$ 
is provided,
- in `Experiments/cost_sensitive_cov_acc.py`
the code of accuracy vs. coverage for cost-sensitive methods is provided,
- in `Experiments/SampleComp.py`
the role of sample complexity is studied, and
- in `Experiments/no_loss_cov_acc.py`
the code of accuracy vs. coverage for methods for 0-1 losses is provided.
