#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file has an example training pipeline from start to finish using the FashionMNIST data."""

import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

sys.path.append("../")
from utilities.training_pipeline import execute_model_optimization

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

# We are using cross entropy here, others are available
loss_fn = nn.CrossEntropyLoss()

# We are using stochastic gradient descent, others are available
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

execute_model_optimization(
    epochs=5,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    early_stop=5,
    save=False,
    model_path="saved_models/sample_model.pth",
    quiet=False,
)
