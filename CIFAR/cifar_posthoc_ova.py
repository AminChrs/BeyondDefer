# Importing modules
import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
import argparse
import os
import shutil
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import json
import sys
sys.path.append('../misc')
import csv 
from losses import Criterion


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cifar_classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# define a function that sets inputs from argparse and print the chose parameters and pass them as keys

def get_args():
    # four parameters of k_line, alpha and cost, and a boolean called is_defer
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--k-line', default=0, type=int, help='line number of k_lists.csv')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha')
    parser.add_argument('--cost', default=0.0, type=float, help='cost')
    parser.add_argument('--is-defer', default=False, action='store_true', help='is_defer')
    # an input integer called cost_idx
    parser.add_argument('--cost-idx', default=0, type=int, help='cost_idx')
    args = parser.parse_args()
    # print values of the parameters
    print("Selected value for alpha:", args.alpha)
    print("Selected value for cost:", args.cost)
    print("Selected list of deterministic labels for the expert:", args.k_line)
    print("Selected value for is_defer:", args.is_defer)
    print("Selected value for cost_idx:", args.cost_idx)
    return args



# print("\n\nSelected value for alpha:", alpha)
# print("Selected value for cost:", cost)
# print("Selected list of deterministic labels for the expert:", k_list)

# Defining the Neural Network
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.softmax = nn.Softmax()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        #out = self.softmax(out)
        return out

# Defining Joint Neural Network
class MetaNet(nn.Module):
    def __init__(self, n_classes, pretrained_model):
        super(MetaNet, self).__init__()
        self.pretrained = pretrained_model
        # removing the last layer (1000 out)
        self.pretrained = nn.Sequential(*list(self.pretrained.children())[:-1])

        self.added_layers = nn.Sequential(nn.Linear(256 + n_classes,100),
                                          nn.ReLU(),
                                          nn.Linear(100, n_classes))
        
    def forward(self, x, m):
        x = self.pretrained(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, 256)
        #print(x.size())
        if type(m) == list:
          one_hot_m = torch.zeros(len(m), 10)
          one_hot_m[torch.arange(len(m)), torch.tensor(m).long()] = 1
          m = one_hot_m.to(device)
        x = self.added_layers(torch.cat((x, m), dim=1))
        #x = self.added_layers(x)
        return x

# Metrics and Utilities

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def metrics_print(net, expert_fn, n_classes, loader):
    '''Computes metrics for deferral

    Parameters
    ----------
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
    correct_pred = {classname: 0 for classname in cifar_classes}
    total_pred = {classname: 0 for classname in cifar_classes}
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size
            exp_prediction = expert_fn(images, labels)
            for i in range(0, batch_size):
                r = (predicted[i].item() == n_classes)
                prediction = predicted[i]
                final_pred = 0
                if predicted[i] == n_classes:
                    max_idx = 0
                    # get second max
                    for j in range(0, n_classes):
                        if outputs.data[i][j] >= outputs.data[i][max_idx]:
                            max_idx = j
                    prediction = max_idx
                else:
                    prediction = predicted[i]
                alone_correct += (prediction == labels[i]).item()
                if r == 0:
                    total += 1
                    final_pred = predicted[i]
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    final_pred = exp_prediction[i]
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1
                if labels[i].item() == final_pred:
                    correct_pred[cifar_classes[labels[i].item()]] += 1
                total_pred[cifar_classes[labels[i].item()]] += 1
    cov = str(total) + str(" out of ") + str(real_total)
    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001),
                "alone classifier": 100 * alone_correct / real_total}
    print(to_print)
    class_accuracies = dict()
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        class_accuracies[classname] = accuracy
        print("Accuracy for class {:5s} is: {:.3f} %".format(classname,
                                                    accuracy))
    return to_print, class_accuracies


def metrics_print_classifier(model, data_loader, defer_net = False, verbose = True):
    '''Computes metrics for classifier

    Parameters
    ----------
    model: model
    data_loader: data loader
    defer_net: boolean to indicate if model is a deferral module (has n_classes +1 outputs)
    verbose: boolean to indicate if the output should be printed or not
    '''
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in cifar_classes}
    total_pred = {classname: 0 for classname in cifar_classes}
    correct = 0
    total = 0
    # again no gradients needed
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1) # maybe no .data
            if defer_net:
                predictions_fixed = predictions
                for i in range(len(predictions_fixed)):
                    if predictions_fixed[i] == 10: #max class
                        max_idx = 0
                        # get second max
                        for j in range(0, 10):
                            if outputs.data[i][j] >= outputs.data[i][max_idx]:
                                max_idx = j
                        prediction = max_idx
                        predictions_fixed[i] = prediction
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[cifar_classes[label]] += 1
                total_pred[cifar_classes[label]] += 1

    total_accuracy = 100 * correct / total
    class_accuracies = dict()
    if verbose:
        print('Accuracy of the network on the %d test batches: %.3f %%' % (total,
            100 * correct / total))
    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        class_accuracies[classname] = accuracy
        if verbose:
            print("Accuracy for class {:5s} is: {:.3f} %".format(classname,
                                                        accuracy))
    return total_accuracy, class_accuracies



def metrics_print_meta(net,expert_fn, n_classes, loader):
    '''Computes metrics for Joint model
    -----
    Arguments:
    net: model
    expert_fn: expert model
    n_classes: number of classes
    loader: data loader
    '''
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in cifar_classes}
    total_pred = {classname: 0 for classname in cifar_classes}
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            m = expert_fn(images, labels)
            m = torch.tensor(m)
            m = m.to(device)

            one_hot_m = torch.zeros((images.size()[0], 10))
            one_hot_m[torch.arange(images.size()[0]), m] = 1
            one_hot_m = one_hot_m.to(device)
            outputs = net(images, one_hot_m)

            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size
            for i in range(0, batch_size):
                prediction = predicted[i]
                total += 1
                correct += (predicted[i] == labels[i]).item()
                if labels[i].item() == prediction:
                    correct_pred[cifar_classes[labels[i].item()]] += 1
                total_pred[cifar_classes[labels[i].item()]] += 1
    acc = 100 * correct / (total)
    print("classifier accuracy:", acc)
    class_accuracies = dict()
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        class_accuracies[classname] = accuracy
        print("Accuracy for class {:5s} is: {:.3f} %".format(classname,
                                                    accuracy))
    return acc, class_accuracies


def metrics_print_human(model, data_loader, expert_fn, k_list, verbose = True):
    '''Computes metrics for human prediction model

    Parameters
    ----------
    model: model
    data_loader: data loader
    expert_fn : expert function
    '''
    # prepare to count predictions for each class
    total_var_class = {classname: 0 for classname in cifar_classes}
    total_pred = {classname: 0 for classname in cifar_classes}
    total_var = 0
    total = 0
    # again no gradients needed
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = F.softmax(model(images))
            m = expert_fn(input, labels)
            m = torch.tensor(m)
            m = m.to(device)

            total += m.size(0)
            # collect the correct predictions for each class
            for label, output in zip(labels, outputs):
                true_dist = torch.zeros(output.size(0)).to(device)
                true_dist[label] = 1
                if label not in k_list:
                    true_dist[:] = 0.1

                var = torch.abs(true_dist - output).sum().item()
                total_var += var
                total_var_class[cifar_classes[label]] += var
                total_pred[cifar_classes[label]] += 1

    total_var = total_var / total
    class_total_var = dict()
    if verbose:
        print('Mean total variation of the network on the %d test batches: %.3f ' % (total,
            total_var))
    
    # print accuracy for each class
    for classname, var in total_var_class.items():
        mean_var = var / total_pred[classname]
        class_total_var[classname] = mean_var
        if verbose:
            print("Mean total variation for class {:5s} is: {:.3f} ".format(classname,
                                                        mean_var))
    return total_var, class_total_var


def metrics_print_posthoc(model_human, model_ai, model_joint, data_loader, expert_fn, cost, verbose=True):
    '''Computes metrics for human prediction model

    Parameters
    ----------
    model_human: human prediction model
    model_ai : classifier
    model_joint : human-ai joint classifier
    data_loader: data loader
    expert_fn : expert prediction function
    cost: cost of querying human (a value between 0 to 1)
    verbost: indicates whether the metrics should be printed or not
    '''
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in cifar_classes}
    total_pred = {classname: 0 for classname in cifar_classes}
    cov_class = {classname: 0 for classname in cifar_classes}
    cov = 0
    correct = 0
    total = 0
    # again no gradients needed
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            m = expert_fn(images, labels)
            m = torch.tensor(m)
            m = m.to(device)

            one_hot_m = torch.zeros((images.size()[0], 10))
            one_hot_m[torch.arange(images.size()[0]), m] = 1
            one_hot_m = one_hot_m.to(device)

            outputs_human = model_human(images)
            outputs_ai = model_ai(images)
            outputs_joint = model_joint(images, one_hot_m)

            #_, predictions_human = torch.max(outputs_human.data, 1) 
            prob_ai , predictions_ai = torch.max(outputs_ai.data, 1) 
            _, predictions_joint = torch.max(outputs_joint.data, 1) 
            prob_posthoc = torch.zeros((m.size(0))).to(device)
            
            for j in range(10):
                one_hot_j = torch.zeros((m.size(0), 10))
                one_hot_j[:, j] = 1
                one_hot_j = one_hot_j.to(device)

                outputs_joint_j = model_joint(images, one_hot_j)
                prob_joint_j, _ = torch.max(outputs_joint_j.data, 1)
                prob_posthoc += outputs_human[:,j] * prob_joint_j

            rejector = prob_posthoc - cost - prob_ai
            predictions = predictions_ai * (rejector <= 0) + predictions_joint * (rejector > 0)
  
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            cov += (rejector <= 0).sum().item()
            # collect the correct predictions for each class
            for label, prediction, rej in zip(labels, predictions, rejector):
                if label == prediction:
                    correct_pred[cifar_classes[label]] += 1
                cov_class[cifar_classes[label]] += (rej <= 0).item()
                total_pred[cifar_classes[label]] += 1

    total_accuracy = 100 * correct / total
    class_accuracies = dict()
    if verbose:
        print('Accuracy of the system on the %d test batches: %.3f %%' % (total,
            100 * correct / total))
        print(f'The coverage of the system is {cov} out of {total}')
    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        class_accuracies[classname] = accuracy
        if verbose:
            print("Accuracy for class {:5s} is: {:.3f} %".format(classname,
                                                        accuracy))
        
    return total_accuracy, class_accuracies, cov, cov_class




# Training Code

## Defining losses
def reject_CrossEntropyLoss(outputs, t_list):
    '''
    The L_{CE} loss implementation for CIFAR
    ----
    outputs: network outputs
    m: cost of deferring to expert cost of classifier predicting (I_{m =y})
    labels: target
    m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
    n_classes: number of classes
    '''
    batch_size = outputs.size()[0]  # batch_size
    outputs = - t_list * torch.log2(outputs)
    # maxt = torch.max(t)
    # outputs = torch.sum((maxt-t)*torch.log2(outputs[range(batch_size), pt.arange(n_classes)]))
    return torch.sum(outputs) / batch_size

def my_CrossEntropyLoss(outputs, labels):
    # Regular Cross entropy loss
    batch_size = outputs.size()[0]  # batch_size
    outputs = - torch.log2(outputs[range(batch_size), labels])  # regular CE
    return torch.sum(outputs) / batch_size


## Defining the train function for one epoch
    
def train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes, alpha, cost, cost_list):
    """Train for one epoch on the training set with deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)

        # get expert  predictions and costs
        batch_size = output.size()[0]  # batch_size
        m = expert_fn(input, target)
        m2 = [0] * batch_size
        t_list = torch.ones((batch_size, n_classes+1)).to(device)
        for j in range(0, batch_size):
            y = int(target[j].item())
            t_list[j] = t_list[j] * cost_list[y]
            t_list[j, y] = 0
            if m[j] == y:
                t_list[j, n_classes] = cost
            else:
                t_list[j, n_classes] += cost
            t_list[j] = torch.max(t_list[j]) - t_list[j] 
            
            if m[j] == y:
                t_list[j, :n_classes] = t_list[j, :n_classes] * alpha
        # done getting expert predictions and costs 
        # compute loss
        criterion = nn.CrossEntropyLoss()
        loss = reject_CrossEntropyLoss(output, t_list)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

def train_reject_class(train_loader, model, optimizer, scheduler, epoch):
    """Train for one epoch on the training set without deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)

        # compute loss
        loss = my_CrossEntropyLoss(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))


def train_reject_meta(train_loader, model, optimizer, scheduler, epoch, expert_fn, n_classes):
    """Train for one epoch on the training set with deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        
        # get expert  predictions and costs
        batch_size = input.size()[0]  # batch_size
        m = expert_fn(input, target)
        m = torch.tensor(m)
        m = m.to(device)

        one_hot_m = torch.zeros((input.size()[0], 10))
        one_hot_m[torch.arange(input.size()[0]), m] = 1
        one_hot_m = one_hot_m.to(device)

        # compute output
        output = model(input, one_hot_m)

        # done getting expert predictions and costs 
        # compute loss
        criterion = nn.CrossEntropyLoss()
        # make a weightlist that is cost_list[i] for Y=i

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))



def train_reject_human(train_loader, model, optimizer, scheduler, epoch, expert_fn):
    """Train for one epoch on the training set without deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        
        # get expert  predictions and costs
        batch_size = input.size()[0]  # batch_size
        m = expert_fn(input, target)
        m = torch.tensor(m)
        m = m.to(device)

        # compute output
        output = model(input)

        # done getting expert predictions and costs 
        # compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, m)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, m, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))


def train_reject_joint(train_loader, model_ai, model_meta, model_human, optimizer, 
                      scheduler, epoch, expert_fn, n_classes):
    """Train for one epoch on the training set with deferral"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_ai = AverageMeter()
    top1_meta = AverageMeter()
    top1_human = AverageMeter()
    
    # switch to train
    model_ai.train()
    model_meta.train()
    model_human.train()
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)
        
        # ai classifier output
        output_ai = model_ai(input)
        
        # meta learner output
        m = expert_fn(input, target)
        m = torch.tensor(m)
        m = m.to(device)

        one_hot_m = torch.zeros((input.size()[0], 10))
        one_hot_m[torch.arange(input.size()[0]), m] = 1
        one_hot_m = one_hot_m.to(device)

        output_meta = model_meta(input, one_hot_m)
        
        # human output
        output_human = model_human(input)
        crit = Criterion()
        loss = crit.comb_ova(output_ai, output_human, output_meta,
                                  m, target, n_classes)
        
        
        # measure accuracy and record loss
        prec_ai = accuracy(output_ai.data, target, topk=(1,))[0]
        prec_human = accuracy(output_human.data, target, topk=(1,))[0]
        prec_meta = accuracy(output_meta.data, target, topk=(1,))[0]
        
        losses.update(loss.data.item(), input.size(0))
        
        top1_ai.update(prec_ai.item(), input.size(0))
        top1_human.update(prec_human.item(), input.size(0))
        top1_meta.update(prec_meta.item(), input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 ai {top1_ai.val:.3f} ({top1_ai.avg:.3f})\t'
                  'Prec@1 human {top1_human.val:.3f} ({top1_human.avg:.3f})\t'
                  'Prec@1 meta {top1_meta.val:.3f} ({top1_meta.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1_ai=top1_ai, top1_human=top1_human, top1_meta=top1_meta))
        

def run_reject_joint(model_ai, model_meta, model_human, expert_fn, 
                     n_dataset, epochs, train_loader, val_loader, 
                     k_list):
    '''
    trains all the classifiers jointly
    model_ai: classifier model
    model_meta: meta learner model
    model_human: human simulator model
    expert_fn: expert function
    n_dataset: number of classes
    epochs: number of epochs
    '''
    # get the number of model parameters
    ## ai
    print('Number of classifier model parameters: {}'.format(
        sum([p.data.nelement() for p in model_ai.parameters()])))
    ## human
    print('Number of human simulator model parameters: {}'.format(
        sum([p.data.nelement() for p in model_human.parameters()])))
    ## meta
    print('Number of meta learner model parameters: {}'.format(
        sum([p.data.nelement() for p in model_meta.parameters()])))
    
    params = list(model_ai.parameters()) + list(model_human.parameters()) + list(model_meta.parameters())
    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(params, 0.001, #0.001
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    
    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)
    
    
    for epoch in range(0, epochs):
        # train for one epoch
        train_reject_joint(train_loader, model_ai, model_meta, model_human, 
                           optimizer, scheduler, epoch, expert_fn, n_dataset)
        if epoch % 10 == 0:
            metrics_print_classifier(model_ai, val_loader)
            metrics_print_human(model_human, val_loader, expert_fn, k_list)
            metrics_print_meta(model_meta, expert_fn, n_dataset, val_loader)
    

class synth_expert:
    '''
    simple class to describe our synthetic expert on CIFAR-10
    ----
    k: number of classes expert can predict
    n_classes: number of classes (10+1 for CIFAR-10)
    '''
    def __init__(self, k_list, n_classes):
        self.k_list = k_list
        self.n_classes = n_classes

    def predict(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i].item() in self.k_list:
                outs[i] = labels[i].item()
            else:
                # change to determinsticly false
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs

def main():
    # checking the device
    print(device)

    args = get_args()
    k_line = args.k_line
    alpha = args.alpha
    cost = args.cost
    cost_idx = args.cost_idx
    is_defer = args.is_defer
    # Reading the inputs
    # k_line = int(sys.argv[1])
    # alpha = float(sys.argv[2])
    # cost = float(sys.argv[3])
    # open cost_lists.csv
    with open('cost_lists.csv', 'r') as file:
        csv_reader = csv.reader(file)
        cost_lists = list(csv_reader)
    # convert the list to ints
    cost_list = [int(x) for x in cost_lists[cost_idx]]
    # cost_list = cost_lists[cost_idx]
    print("cost_list", cost_list)
    with open('k_lists.csv', 'r') as file:
        csv_reader = csv.reader(file)
        k_lists = list(csv_reader)

    k_list = k_lists[k_line]
    k_list = set(map(lambda x: int(x), k_list))


    # Data
    use_data_aug = False
    n_dataset = 10  # cifar-10



    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if use_data_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if n_dataset == 10:
        dataset = 'cifar10'
    elif n_dataset == 100:
        dataset = 'cifar100'

    kwargs = {'num_workers': 0, 'pin_memory': True}


    train_dataset_all = datasets.__dict__[dataset.upper()]('../data', train=True, download=True,
                                                            transform=transform_train)
    train_size = int(0.90 * len(train_dataset_all))
    test_size = len(train_dataset_all) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset_all, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=128, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=128, shuffle=True, **kwargs)


    # Initialize Expert and Model
    k = 5 # number of classes expert can predict
    n_dataset = 10
    expert = synth_expert(k_list , n_dataset)

    ## calculating expert's accuracy
    total = len(test_dataset)
    correct = 0
    for data in test_dataset:
        input, y = data[0], data[1]
        exp_pred = expert.predict(input, torch.tensor([y]))
        if exp_pred[0] == y:
            correct += 1
    acc = correct/total * 100
    print(f'The expert accuracy on test dataset is {acc:.2f}')

    results = dict()
    results['input'] = {'k_list': list(k_list), 'alpha': alpha, 'cost': cost}

    # Training the models
    ## deferral
    if (is_defer):
        print(f"\n\n----------- Training the deferral with alpha = {alpha} ----------\n\n")
        model_deferral = WideResNet(10, n_dataset + 1, 4, dropRate=0).to(device) # fancy
        run_reject(model_deferral, n_dataset, expert.predict, 200, alpha, train_loader, val_loader, cost, cost_list) # train for 200 epochs

        print("\n\n----------- Printing the metrics of the deferral ----------\n\n")
        metrics_deferral, class_accuracies_deferral = metrics_print(model_deferral, expert.predict, n_dataset, val_loader)
        results['deferral'] = {'metrics': metrics_deferral, 'class accuracies': class_accuracies_deferral}
    if (not is_defer):
         ## human
        print(f"\n\n----------- Training the all classifiers jointly ----------\n\n")
        model_human = WideResNet(10, n_dataset, 4, dropRate=0).to(device) 
        model_meta = MetaNet(10, WideResNet(10, n_dataset + 1, 4, dropRate=0))
        model_ai = WideResNet(10, n_dataset + 1, 4, dropRate=0).to(device) 
        
        run_reject_joint(model_ai, model_meta, model_human, expert.predict, 10, 200, train_loader, val_loader, k_list)
        
        print("\n\n----------- Printing the metrics of the classifier ----------\n\n")
        accuracy_ai, class_accuracies_ai = metrics_print_classifier(model_ai, val_loader)
        results['ai'] = {'accuracy':accuracy_ai, 'class accuracies': class_accuracies_ai}
        

        print("\n\n----------- Printing the metrics of the joint classifier ----------\n\n")
        accuracy_joint, class_accuracies_joint = metrics_print_meta(model_meta, expert.predict, n_dataset, val_loader)
        results['joint'] = {'accuracy': accuracy_joint, 'class accuracies': class_accuracies_joint}

       
        print("\n\n----------- Printing the metrics of the human predictor ----------\n\n")
        total_var_human, class_total_vars_human = metrics_print_human(model_human, val_loader, expert.predict, k_list)
        results['human'] = {'total variation': total_var_human, 'class total variation': class_total_vars_human}

        ## hybrid
        print("\n\n----------- Printing the metrics of the hybrid predictor ----------\n\n")
        total_accuracy_hybrid, class_accuracies_hybrid, cov, cov_class = metrics_print_posthoc(model_human, model_ai, model_meta, val_loader, expert.predict, cost)
        results['hybrid'] = {'accuracy': total_accuracy_hybrid, 'class accuracy': class_accuracies_hybrid, 'coverage': cov, 'class coverage': cov_class}

    if (is_defer):
        filename = 'files/results_is_defer='+str(is_defer)+'_k=' + str(k_line) + '_a=' + str(alpha) + '_c=' + str(cost) + '.json' 
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        filename = 'files/results_is_defer='+str(is_defer)+'_k=' + str(k_line) + '_c=' + str(cost) + '.json' 
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
if __name__ == '__main__':
    main()