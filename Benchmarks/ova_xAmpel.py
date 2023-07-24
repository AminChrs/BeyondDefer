# In this file, I generate a two-dimensional input from a Gaussian mixture model
# where the true label is the index of the Gaussian and the human is correct for
# the first Gaussian distribution but not for the second one. Then I minimize
# the loss function defined in OvA package

import sys
sys.path.append('../misc')
from losses import Criterion
import numpy as np
import torch as pt



def permute(X, Y, M):
    perm = np.random.permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    M = M[perm]
    return X, Y, M

def mix_Gauss(mu1, mu2, var, n):
    X1 = np.random.normal(mu1, var, n)
    X2 = np.random.normal(mu2, var, n)
    X = np.concatenate((X1, X2))
    Y = np.concatenate((np.zeros(n), np.ones(n)))
    M = np.concatenate((np.zeros(n), np.random.randint(2, size=n)))
    X, Y, M = permute(X, Y, M)

# NN class
class Classifier(pt.nn.Module):
    def __init__(self, n_in, n_out):
        super(Classifier, self).__init__()
        self.fc1 = pt.nn.Linear(n_in, 10)
        self.fc2 = pt.nn.Linear(10, n_out)

    def forward(self, x):
        x = pt.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

epochs = 100
batch_size = 100
X, Y, M = mix_Gauss(0, 1, 1, 1000)
num_batches = X.shape[0] // batch_size

classifier = Classifier(2, 2)
simulator = Classifier(2, 2)
meta_classifier = Classifier(4, 2)

# training loop
for epoch in range(epochs):
    for i in range(num_batches):
        x = pt.tensor(X[i*batch_size:(i+1)*batch_size], dtype=pt.float32)
        y = pt.tensor(Y[i*batch_size:(i+1)*batch_size], dtype=pt.long)
        m = pt.tensor(M[i*batch_size:(i+1)*batch_size], dtype=pt.long)
        optimizer = pt.optim.SGD(classifier.parameters(), lr=0.01)
        optimizer.zero_grad()
        outputs_classifier = classifier(x)
        outputs_simulator = simulator(x)
        meta_outputs = meta_classifier(pt.cat((x, m), dim=1))
        comb_ova = Criterion().comb_ova
        loss = comb_ova(outputs_classifier, simulator(x), meta_classifier(pt.cat((outputs_classifier, simulator(x)), dim=1)), m, y, 2)
        loss.backward()
        optimizer.step()