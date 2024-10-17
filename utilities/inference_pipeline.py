#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file contains all methods used for inference pipelines."""

import torch


def infer_image(
    input_data, model, classes: list[str], device, transform=None
) -> dict[str, float]:
    """
    Method to infer a single image of the PIL Image format using a PyTorch model.

    Args:
        input_data (PIL.Image): This should be the image to infer.
        model (nn.Module): The model to use during inference.
        classes (list[str]): The classes to match the output with.
        device (str): The device to run the inference on.
        transform (Torch Transform, optional): The preprocessing transform to apply

    Returns:
        dict[str, float]: The mapped dictionary from classes to output values (softmax applied)
    """
    model.to(device)
    model.eval()

    if transform:
        input_data.transfrom

    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data)

    if input_data.dim() == 3:  # assuming input is a single image (C, H, W)
        input_data = input_data.unsqueeze(0)  # Making room for channel dim

    input_data.to(device)

    inferences = {}

    # Using the model loaded in the previous step
    model.eval()

    # Bypassing gradient similar to validation
    with torch.no_grad():
        pred = model(input_data)
        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, _class in enumerate(classes):
            inferences[_class] = pred[i]
    return inferences
