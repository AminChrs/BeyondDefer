import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
# import copy
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MetaNet(nn.Module):

    def __init__(self, n_classes, pretrained_model, dim_last_layer,
                 n_conv_layers=0,
                 n_linear_layers=0,
                 n_linear_layers_after_concat=1,
                 kernel_sizes_conv=[], width_linear=[],
                 width_linear_after_concat=[],
                 remove_layers=[]):
        super(MetaNet, self).__init__()
        self.pretrained = pretrained_model

        for layer in remove_layers:
            if "." not in layer:
                setattr(self.pretrained, layer, Identity())
            else:
                layer = layer.split(".")
                inner_layer = self.pretrained
                for i in range(len(layer) - 1):
                    inner_layer = getattr(inner_layer, layer[i])
                setattr(inner_layer, layer[-1], Identity())

        self.n_linear_layers = n_linear_layers
        self.conv_layers = []
        self.linear_layers = []
        self.linear_layers_after_concat = []

        for i in range(n_conv_layers):
            if i == 0:
                self.conv_layers.append(nn.Conv2d(dim_last_layer[0],
                                                  dim_last_layer[0],
                                                  kernel_sizes_conv[i]))
                shape_last = ((dim_last_layer[1:] - kernel_sizes_conv[i]) / 2)\
                    + 1
            else:
                self.conv_layers.append(nn.Conv2d(dim_last_layer[0],
                                                  dim_last_layer[0],
                                                  kernel_sizes_conv[i]))
                shape_last = ((shape_last - kernel_sizes_conv[i]) / 2) + 1

        if n_conv_layers == 0:
            shape_last = dim_last_layer[1:]

        self.linear_first_size = shape_last[0] * shape_last[1] * \
            dim_last_layer[0]

        for i in range(n_linear_layers):
            if i == 0:
                self.linear_layers.append(nn.Linear(self.linear_first_size,
                                                    width_linear[i]))
            else:
                self.linear_layers.append(nn.Linear(width_linear[i-1],
                                                    width_linear[i]))

        if n_linear_layers == 0:
            width_linear = [shape_last[0] * shape_last[1] * dim_last_layer[0]]
            self.linear_layers = [Identity()]

        for i in range(n_linear_layers_after_concat):
            if i == 0 and i != n_linear_layers_after_concat - 1:
                self.linear_layers_after_concat.append(
                    nn.Sequential(
                        nn.Linear(n_classes + width_linear[-1],
                                  width_linear_after_concat[i]),
                        nn.ReLU()
                    )
                )
            elif i != n_linear_layers_after_concat - 1:
                self.linear_layers_after_concat.append(
                    nn.Sequential(
                        nn.Linear(width_linear_after_concat[i-1],
                                  width_linear_after_concat[i]),
                        nn.ReLU()
                    )
                )
            elif i != 0:
                self.linear_layers_after_concat.append(
                        nn.Linear(width_linear_after_concat[i], n_classes)
                )
            else:
                self.linear_layers_after_concat.append(
                    nn.Sequential(
                        nn.Linear(n_classes + width_linear[-1],
                                  n_classes),
                    )
                )

        if n_linear_layers_after_concat == 0:
            self.added_layers = nn.Linear(n_classes + width_linear[-1],
                                          n_classes)
        else:
            self.added_layers = nn.Sequential(nn.Linear(256 + n_classes,
                                                        100),
                                              nn.ReLU(),
                                              nn.Linear(100, n_classes))

    def forward(self, x, m):
        x = self.pretrained(x)
        if (len(x.shape) == 4):
            x = F.avg_pool2d(x, 8)
        for layer in self.conv_layers:
            x = layer(x)
            x = F.relu(x)
        x = x.view(-1, self.linear_first_size)
        for layer in self.linear_layers:
            x = layer(x)
            x = F.relu(x)
        if type(m) == list:
            one_hot_m = torch.zeros(len(m), 10)
            one_hot_m[torch.arange(len(m)), torch.tensor(m).long()] = 1
            m = one_hot_m.to(device)
        x = torch.cat((x, m), dim=1)
        for layer in self.linear_layers_after_concat:
            x = layer(x)
        return x

    def to(self, dev):
        self.pretrained.to(dev)
        for layer in self.conv_layers:
            layer.to(dev)
        for layer in self.linear_layers:
            layer.to(dev)
        for layer in self.linear_layers_after_concat:
            layer.to(dev)
        self.added_layers.to(dev)
        return self

    def parameters(self):
        list = [*self.pretrained.parameters()]
        if len(self.conv_layers) != 0:
            for layer in self.conv_layers:
                list += layer.parameters()
        if self.n_linear_layers != 0:
            for layer in self.linear_layers:
                list += layer.parameters()
        if len(self.linear_layers_after_concat) != 0:
            for layer in self.linear_layers_after_concat:
                list += layer.parameters()
        if len(self.added_layers) != 0:
            list += self.added_layers.parameters()
        return list

    def weight_init(self):
        for param in self.parameters():
            if param.requires_grad:
                param.data.uniform_(-1e-5, 1e-5)
