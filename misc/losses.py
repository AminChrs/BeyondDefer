import torch.nn as nn
import torch
import numpy as np




class Criterion(object):
	def __init__(self):
		pass
		
	def softmax(self, outputs, m, labels, m2, n_classes):
		'''
		The L_{CE} loss implementation for CIFAR
		----
		outputs: network outputs
		m: cost of deferring to expert cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
		labels: target
		m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
		n_classes: number of classes
		'''
		batch_size = outputs.size()[0]  # batch_size
		rc = [n_classes] * batch_size
		outputs = -m * torch.log2(outputs[range(batch_size), rc]) - m2 * torch.log2(
			outputs[range(batch_size), labels])  
		return torch.sum(outputs) / batch_size

	def ova(self, outputs, m, labels, m2, n_classes):
		batch_size = outputs.size()[0]
		l1 = Criterion.LogisticLoss(outputs[range(batch_size), labels], 1)
		l2 = torch.sum(Criterion.LogisticLoss(outputs[:,:n_classes], -1), dim=1) - Criterion.LogisticLoss(outputs[range(batch_size),labels],-1)
		l3 = Criterion.LogisticLoss(outputs[range(batch_size), n_classes], -1)
		l4 = Criterion.LogisticLoss(outputs[range(batch_size), n_classes], 1)

		l5 = m * (l4 - l3)

		l = m2 * (l1 + l2) + l3 + l5

		return torch.mean(l)
	def comb_ova(self, outputs_classifier, outputs_sim, outputs_meta, m, labels, n_classes):
		'''
		The L_{comb} loss implementation.
		----
		outputs: network outputs
		m: human label
		labels: target
		outputs_classifier: output of the classifier
		outputs_sim: output of the human simulator
		outputs_meta: output of the meta classifier
		n_classes: number of classes
		'''
		batch_size = outputs_classifier.size()[0]  # batch_size
		l1 = Criterion.LogisticLoss(outputs_classifier[range(batch_size), labels], 1)
		l2 = torch.sum(Criterion.LogisticLoss(outputs_classifier[:,:n_classes], -1), dim=1) - Criterion.LogisticLoss(outputs_classifier[range(batch_size),labels],-1)
		l3 = Criterion.LogisticLoss(outputs_meta[range(batch_size), labels], 1)
		l4 = torch.sum(Criterion.LogisticLoss(outputs_meta[:,:n_classes], -1), dim=1) - Criterion.LogisticLoss(outputs_meta[range(batch_size),labels],-1)
		l5 = Criterion.LogisticLoss(outputs_sim[range(batch_size), m], 1)
		l6 = torch.sum(Criterion.LogisticLoss(outputs_sim[:,:n_classes], -1), dim=1) - Criterion.LogisticLoss(outputs_sim[range(batch_size),m],-1)
		l = l1 + l2 + l3 + l4 + l5 + l6
		return torch.mean(l)

	@staticmethod
	def LogisticLoss(outputs, y):
		outputs[torch.where(outputs==0.0)] = (-1*y)*(-1*np.inf)
		l = torch.log2(1 + torch.exp((-1*y)*outputs))
		return l