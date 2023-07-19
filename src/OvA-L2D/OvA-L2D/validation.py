# Analyze the confidences on test data

from __future__ import division
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import argparse
import json
import os
from collections import defaultdict
import numpy as np
from scipy import stats
import torch
import random

from data_utils import cifar
from losses.losses import *
from models.experts import synth_expert
from models.wideresnet import WideResNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)


def evaluate(model,
			expert_fn,
			loss_fn,
			n_classes,
			data_loader,
			config,
			deferral_cost=0.0):
	'''
	Computes metrics for deferal
	-----
	Arguments:
	net: model
	expert_fn: expert model
	n_classes: number of classes
	loader: data loader
	'''
	correct = 0
	correct_sys = 0
	exp = 0
	exp_total = 0
	total = 0
	real_total = 0
	alone_correct = 0
	alpha = config["alpha"]
	losses = []
	with torch.no_grad():
		for data in data_loader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			if config["loss_type"] == "softmax":
				outputs = F.softmax(outputs, dim=1)
				outputs = outputs / (torch.sum(outputs[:,:n_classes], dim=1).view(-1,1))
			if config["loss_type"] == "ova":
				ouputs = F.sigmoid(outputs)

			outputs[:,:n_classes] = outputs[:, :n_classes] + deferral_cost
			_, predicted = torch.max(outputs.data, 1)
			batch_size = outputs.size()[0]  # batch_size
			exp_prediction = expert_fn(images, labels)

			m = [0]*batch_size
			m2 = [0] * batch_size
			for j in range(0, batch_size):
				if exp_prediction[j] == labels[j][0].item():
					m[j] = 1
					m2[j] = alpha
				else:
					m[j] = 0
					m2[j] = 1

			m = torch.tensor(m)
			m2 = torch.tensor(m2)
			m = m.to(device)
			m2 = m2.to(device)

			loss = loss_fn(outputs, m, labels[:,0], m2, n_classes)
			losses.append(loss.item())

			for i in range(0, batch_size):
				r = (predicted[i].item() == n_classes)
				prediction = predicted[i]
				if predicted[i] == n_classes:
					max_idx = 0
					# get second max
					for j in range(0, n_classes):
						if outputs.data[i][j] >= outputs.data[i][max_idx]:
							max_idx = j
					prediction = max_idx
				else:
					prediction = predicted[i]
				alone_correct += (prediction == labels[i][0]).item()
				if r == 0:
					total += 1
					correct += (predicted[i] == labels[i][0]).item()
					correct_sys += (predicted[i] == labels[i][0]).item()
				if r == 1:
					exp += (exp_prediction[i] == labels[i][0].item())
					correct_sys += (exp_prediction[i] == labels[i][0].item())
					exp_total += 1
				real_total += 1
	cov = str(total) + str(" out of") + str(real_total)
	to_print = {"coverage": cov, "system_accuracy": 100 * correct_sys / real_total,
				"expert_accuracy": 100 * exp / (exp_total + 0.0002),
				"classifier_accuracy": 100 * correct / (total + 0.0001),
				"alone_classifier": 100 * alone_correct / real_total,
				"validation_loss": np.average(losses)}
	print(to_print, flush=True)
	return to_print

def forward(model, dataloader, expert_fns, n_classes, n_experts):
	confidence = []
	true = []
	expert_predictions = defaultdict(list)
	# density = []

	with torch.no_grad():
		for inp, lbl in dataloader:
			inp = inp.to(device)
			conf = model(inp)
			for i, fn in enumerate(expert_fns):
				expert_pred1 = fn(inp, lbl)
				expert_predictions[i].append(expert_pred1)
			confidence.append(conf.cpu())
			true.append(lbl[:, 0])


	true = torch.stack(true, dim=0).view(-1)
	confidence = torch.stack(confidence, dim=0).view(-1, n_classes + n_experts)
	for k, v in expert_predictions.items():
		expert_predictions[k] = torch.stack([torch.tensor(k) for k in v], dim=0).view(-1)

	print(true.shape, confidence.shape, [v.shape for k, v in
										 expert_predictions.items()])  # ,expert_predictions1.shape, expert_predictions2.shape) #, density.shape)
	return true, confidence, [v.numpy() for k, v in
							  expert_predictions.items()]  # (expert_predictions1, expert_predictions2) #, density


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


def validation(model_name, expert_fns, config, seed="", cost=0.0):
	def filter(dict_):
		d = {}
		for k, v in dict_.items():
			if torch.is_tensor(v):
				v = v.item()
			d[k] = v
		return d

	def get(severity, dl):
		# true, confidence, expert_predictions = forward(model, dl, expert_fns, n_dataset, n_expert)
		# print("shapes: true labels {}, confidences {}, expert_predictions {}".format(\
		# 	true.shape, confidence.shape, np.array(expert_predictions).shape))

		criterion = Criterion()
		loss_fn = getattr(criterion, config["loss_type"])
		n_classes = n_dataset
		print("Evaluate...")
		result_ = evaluate(model, expert_fns[0], loss_fn, n_classes, dl, config, deferral_cost=cost)
		# true_label[severity] = true.numpy()
		# classifier_confidence[severity] = confidence.numpy()
		# expert_preds[severity] = expert_predictions
		result[severity] = filter(result_)

	result = {}
	classifier_confidence = {}
	true_label = {}
	expert_preds = {}

	n_dataset = 10
	batch_size = 1024
	n_expert = len(expert_fns)

	# Data ===
	ood_d, test_d = cifar.read(severity=0, slice_=-1, test=True, only_id=True)

	kwargs = {'num_workers': 1, 'pin_memory': True}
	test_dl = torch.utils.data.DataLoader(test_d, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

	# Model ===
	model = WideResNet(28, 3, n_dataset + n_expert, 4, dropRate=0.0)
	model_path = os.path.join(config["ckp_dir"], config["experiment_name"] + '.pt')
	model.load_state_dict(torch.load(model_path, map_location=device))
	model = model.to(device)

	get('test', test_dl)

	# with open(config["ckp_dir"] + 'true_label_multiple_experts_new' + model_name + '.txt', 'w') as f:
	# 	json.dump(json.dumps(true_label, cls=NumpyEncoder), f)

	# with open(config["ckp_dir"] + 'confidence_multiple_experts_new' + model_name + '.txt', 'w') as f:
	# 	json.dump(json.dumps(classifier_confidence, cls=NumpyEncoder), f)

	# with open(config["ckp_dir"] + 'expert_predictions_multiple_experts_new' + model_name + '.txt', 'w') as f:
	# 	json.dump(json.dumps(expert_preds, cls=NumpyEncoder), f)

	# with open(config["ckp_dir"] + 'validation_multiple_experts_new' + model_name + '.txt', 'w') as f:
	# 	json.dump(json.dumps(result, cls=NumpyEncoder), f)
	return result


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--batch_size", type=int, default=1024)
	parser.add_argument("--alpha", type=float, default=1.0,
						help="scaling parameter for the loss function, default=1.0.")
	parser.add_argument("--expert_type", type=str, default="predict",
						help="specify the expert type. For the type of experts available, see-> models -> experts. defualt=predict.")
	parser.add_argument("--n_classes", type=int, default=10,
						help="K for K class classification.")
	parser.add_argument("--k", type=int, default=5)
	parser.add_argument("--loss_type", type=str, default="ova",
						help="surrogate loss type for learning to defer.")
	parser.add_argument("--ckp_dir", type=str, default="./Models",
						help="directory name to save the checkpoints.")
	parser.add_argument("--experiment_name", type=str, default="default",
						help="specify the experiment name. Checkpoints will be saved with this name.")

	config = parser.parse_args().__dict__
	config["ckp_dir"] = config["ckp_dir"] + "/" + config["loss_type"]
	#print(config)

	alpha = 1.0
	n_dataset = 10
	seeds = [''] #948, 625, 436]
	accuracy = []

	for seed in seeds:
		print("seed is {}".format(seed))
		if seed != '':
			set_seed(seed)
		acc = []
		for deferral_cost in [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
			print("deferral cost is {}".format(deferral_cost))
			model_name = config["experiment_name"]
			# Expert ===
			expert = synth_expert(config["k"], config["n_classes"])
			expert_fn = getattr(expert, config["expert_type"])


			result = validation(model_name, [expert_fn], config, seed=seed, cost=deferral_cost)
			acc.append(result["test"]["system_accuracy"])
			accuracy.append(acc)

	print("===Mean and Standard Error===")
	print(np.mean(np.array(accuracy), axis=0))
	print(stats.sem(np.array(accuracy), axis=0))

