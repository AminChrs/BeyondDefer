import torch
import torch.nn as nn
from human_ai_defer.datasetsdefer.cifar_synth import CifarSynthDataset
from human_ai_defer.datasetsdefer.hatespeech import HateSpeech
from MyNet.networks import MetaNet
from human_ai_defer.networks.cnn import WideResNet
from human_ai_defer.baselines.selective_prediction import\
     SelectivePrediction
from human_ai_defer.networks.cnn import NetSimple
from human_ai_defer.networks.cnn import DenseNet121_CE
from human_ai_defer.networks.linear_net import LinearNet
import os
import numpy as np
import logging


def optimizer_scheduler():

    def scheduler(z, length):
        return torch.optim.lr_scheduler.CosineAnnealingLR(z, length)

    def optimizer(params, lr): return torch.optim.Adam(params, lr=lr,
                                                       )
    return optimizer, scheduler


def load_model_trained_cifarh(n, device, path):
    model = WideResNet(28, 10, 4, dropRate=0).to(device)
    model.load_state_dict(torch.load(path))
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(50, n).to(device)
    return model


def model_cifarh(n, device, path):
    if os.path.exists(path):
        model = load_model_trained_cifarh(n, device, path)
    else:
        model = WideResNet(28, 10, 4, dropRate=0).to(device)
        optimizer, scheduler = optimizer_scheduler()
        dataset = CifarSynthDataset(0, True, batch_size=512)
        SP = SelectivePrediction(model, device, plotting_interval=10)
        SP.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=200,
            optimizer=optimizer,
            lr=0.001,
            verbose=True,
            scheduler=scheduler
        )
        torch.save(model.state_dict(), path)
        model = load_model_trained_cifarh(n, device, path)
    return model


def model_imagenet(n, device):
    model_linear = DenseNet121_CE(n).to(device)
    for param in model_linear.parameters():
        param.requires_grad = False
    model_linear.densenet121.classifier.requires_grad_(True)
    return model_linear


def networks(dataset_name, method, device):
    if dataset_name == "cifar_synth":
        if method == "BD":
            model_classifier = NetSimple(10, 50, 50, 100, 20).to(device)
            model_human = NetSimple(10, 50, 50, 100, 20).to(device)
            model_meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20),
                                 [1, 20, 1],
                                 remove_layers=["fc3", "softmax"]).to(device)
            return model_classifier, model_human, model_meta
        if method == "Additional":
            model_classifier = NetSimple(11, 50, 50, 100, 20).to(device)
            model_human = NetSimple(10, 50, 50, 100, 20).to(device)
            model_meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20),
                                 [1, 20, 1],
                                 remove_layers=["fc3", "softmax"]).to(device)
            return model_classifier, model_human, model_meta
        if method == "LearnedBeyond":
            model_classifier = NetSimple(11, 50, 50, 100, 20).to(device)
            model_meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20),
                                 [1, 20, 1],
                                 remove_layers=["fc3", "softmax"]).to(device)
            return model_classifier, model_meta
        if method == "LearnedAdditional":
            model_classifier = NetSimple(12, 50, 50, 100, 20).to(device)
            model_meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20),
                                 [1, 20, 1],
                                 remove_layers=["fc3", "softmax"]).to(device)
            return model_classifier, model_meta
        if method == "CompConfMeta":
            model_classifier = NetSimple(10, 50, 50, 100, 20).to(device)
            model_meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20),
                                 [1, 20, 1],
                                 remove_layers=["fc3", "softmax"]).to(device)
            model_defer = NetSimple(2, 50, 50, 100, 20).to(device)
            model_defer_meta = NetSimple(2, 50, 50, 100, 20).to(device)
            return model_classifier, model_meta, model_defer, model_defer_meta
        elif method == "AFE":
            model_classifier = NetSimple(10, 50, 50, 100, 20).to(device)
            model_meta = MetaNet(10, NetSimple(10, 50, 50, 100, 20),
                                 [1, 20, 1],
                                 remove_layers=["fc3", "softmax"]).to(device)
            return model_classifier, model_meta
        elif method == "triage" or method == "confidence":
            model_class = NetSimple(10, 50, 50, 100, 20).to(device)
            model_expert = NetSimple(2, 50, 50, 100, 20).to(device)
            return model_class, model_expert
        elif method == "selective":
            model = NetSimple(10, 50, 50, 100, 20).to(device)
            return model
        else:
            model = NetSimple(11, 50, 50, 100, 20).to(device)
            return model
    elif dataset_name == "cifar_10h":
        if method == "BD":
            model_classifier = model_cifarh(10, device,
                                            "./models/cifar10h.pt")
            model_human = model_cifarh(10, device,
                                       "./models/cifar10h.pt")
            model_meta = MetaNet(10, model_cifarh(10, device,
                                                  "./models/cifar10h.pt"),
                                 [1, 50, 1],
                                 remove_layers=["fc2", "softmax"]).to(device)
            return model_classifier, model_human, model_meta
        if method == "Additional":
            model_classifier = model_cifarh(11, device,
                                            "./models/cifar10h.pt")
            model_human = model_cifarh(10, device,
                                       "./models/cifar10h.pt")
            model_meta = MetaNet(10, model_cifarh(10, device,
                                                  "./models/cifar10h.pt"),
                                 [1, 50, 1],
                                 remove_layers=["fc2", "softmax"]).to(device)
            return model_classifier, model_human, model_meta
        if method == "LearnedBeyond":
            model_classifier = model_cifarh(11, device,
                                            "./models/cifar10h.pt")
            model_meta = MetaNet(10, model_cifarh(10, device,
                                                  "./models/cifar10h.pt"),
                                 [1, 50, 1],
                                 remove_layers=["fc2", "softmax"]).to(device)
            return model_classifier, model_meta
        if method == "LearnedAdditional":
            model_classifier = model_cifarh(12, device,
                                            "./models/cifar10h.pt")
            model_meta = MetaNet(10, model_cifarh(10, device,
                                                  "./models/cifar10h.pt"),
                                 [1, 50, 1],
                                 remove_layers=["fc2", "softmax"]).to(device)
            return model_classifier, model_meta
        if method == "CompConfMeta":
            model_classifier = model_cifarh(10, device,
                                            "./models/cifar10h.pt")
            model_meta = MetaNet(10, model_cifarh(10, device,
                                                  "./models/cifar10h.pt"),
                                 [1, 50, 1],
                                 remove_layers=["fc2", "softmax"]).to(device)
            model_defer = model_cifarh(2, device,
                                       "./models/cifar10h.pt")
            model_defer_meta = model_cifarh(2, device,
                                            "./models/cifar10h.pt")
            return model_classifier, model_meta, model_defer, model_defer_meta
        elif method == "AFE":
            model_classifier = model_cifarh(10, device,
                                            "./models/cifar10h.pt")
            model_meta = MetaNet(10, model_cifarh(10, device, 
                                                  "./models/cifar10h.pt"),
                                 [1, 50, 1],
                                 remove_layers=["fc2", "softmax"]).to(device)
            logging.info("AFE model assigned!")
            return model_classifier, model_meta
        elif method == "triage" or method == "confidence":
            model_classifier = model_cifarh(10, device,
                                            "./models/cifar10h.pt")
            model_expert = model_cifarh(2, device,
                                        "./models/cifar10h.pt")
            return model_classifier, model_expert
        elif method == "selective":
            model = model_cifarh(10, device, "./models/cifar10h.pt")
            return model
        else:
            model = model_cifarh(11, device, "./models/cifar10h.pt")
            return model
    elif dataset_name == "imagenet":
        if method == "BD":
            model_classifier = model_imagenet(16, device)
            model_human = model_imagenet(16, device)
            model_meta = MetaNet(16, model_imagenet(16, device),
                                 [1, 1024, 1],
                                 remove_layers=["densenet121.classifier"]
                                 ).to(device)
            return model_classifier, model_human, model_meta
        if method == "Additional":
            model_classifier = model_imagenet(17, device)
            model_human = model_imagenet(16, device)
            model_meta = MetaNet(16, model_imagenet(16, device),
                                 [1, 1024, 1],
                                 remove_layers=["densenet121.classifier"]
                                 ).to(device)
            return model_classifier, model_human, model_meta
        if method == "LearnedBeyond":
            model_classifier = model_imagenet(17, device)
            model_meta = MetaNet(16, model_imagenet(16, device),
                                 [1, 1024, 1],
                                 remove_layers=["densenet121.classifier"]
                                 ).to(device)
            return model_classifier, model_meta
        if method == "LearnedAdditional":
            model_classifier = model_imagenet(18, device)
            model_meta = MetaNet(16, model_imagenet(16, device),
                                 [1, 1024, 1],
                                 remove_layers=["densenet121.classifier"]
                                 ).to(device)
            return model_classifier, model_meta
        if method == "CompConfMeta":
            model_classifier = model_imagenet(16, device)
            model_meta = MetaNet(16, model_imagenet(16, device),
                                 [1, 1024, 1],
                                 remove_layers=["densenet121.classifier"]
                                 ).to(device)
            model_defer = model_imagenet(2, device)
            model_defer_meta = model_imagenet(2, device)
            return model_classifier, model_meta, model_defer, model_defer_meta
        elif method == "AFE":
            model_classifier = model_imagenet(16, device)
            model_meta = MetaNet(16, model_imagenet(16, device),
                                 [1, 1024, 1],
                                 remove_layers=["densenet121.classifier"]
                                 ).to(device)
            return model_classifier, model_meta
        elif method == "triage" or method == "confidence":
            model_classifier = model_imagenet(16, device)
            model_expert = model_imagenet(2, device)
            return model_classifier, model_expert
        elif method == "selective":
            model = model_imagenet(16, device)
            return model
        else:
            model = model_imagenet(17, device)
            return model
    elif dataset_name == "hatespeech":
        if os.path.exists("./data/hatespeech-dim.npz"):
            dim = np.load("./data/hatespeech-dim.npz")
            d = dim["d"]
        else:
            dataset = HateSpeech("./data", True, False,
                                 'random_annotator', device)
            d = dataset.d

            np.savez("./data/hatespeech-dim.npz", d=d)

        if method == "BD":
            model_classifier = LinearNet(d, 3).to(device)
            model_human = LinearNet(d, 3).to(device)
            model_meta = MetaNet(3, LinearNet(d, 10),
                                 [1, 10, 1]).to(device)
            return model_classifier, model_human, model_meta
        if method == "Additional":
            model_classifier = LinearNet(d, 4).to(device)
            model_human = LinearNet(d, 3).to(device)
            model_meta = MetaNet(3, LinearNet(d, 10),
                                 [1, 10, 1]).to(device)
            return model_classifier, model_human, model_meta
        if method == "LearnedBeyond":
            model_classifier = LinearNet(d, 4).to(device)
            model_meta = MetaNet(3, LinearNet(d, 10),
                                 [1, 10, 1]).to(device)
            return model_classifier, model_meta
        if method == "LearnedAdditional":
            model_classifier = LinearNet(d, 5).to(device)
            model_meta = MetaNet(3, LinearNet(d, 10),
                                 [1, 10, 1]).to(device)
            return model_classifier, model_meta
        if method == "CompConfMeta":
            model_classifier = LinearNet(d, 3).to(device)
            model_meta = MetaNet(3, LinearNet(d, 10),
                                 [1, 10, 1]).to(device)
            model_defer = LinearNet(d, 2).to(device)
            model_defer_meta = LinearNet(d, 2).to(device)
            return model_classifier, model_meta, model_defer, model_defer_meta
        elif method == "AFE":
            model_classifier = LinearNet(d, 3).to(device)
            model_meta = MetaNet(3, LinearNet(d, 10),
                                 [1, 10, 1]).to(device)
            return model_classifier, model_meta
        elif method == "triage" or method == "confidence":
            model_classifier = LinearNet(d, 3).to(device)
            model_expert = LinearNet(d, 2).to(device)
            return model_classifier, model_expert
        elif method == "selective":
            model = LinearNet(d, 3).to(device)
            return model
        else:
            model = LinearNet(d, 4).to(device)
            return model

