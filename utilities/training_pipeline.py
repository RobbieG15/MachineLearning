#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file contains all methods used for training pipelines."""

import time

import torch


def execute_model_optimization(
    epochs,
    train_dataloader,
    test_dataloader,
    model,
    loss_fn,
    optimizer,
    device,
    early_stop=None,
    save=True,
    model_path="models/model.pth",
    test_type="regular",
    quiet=True,
):
    """
    Optimizes a model by running training and test pipelines a certain number of times.

    Args:
        epochs (int): The number of times to run through the loop.
        train_dataloader (torch.utils.data.DataLoader): The dataloader to hold the training data.
        test_dataloader (torch.utils.data.DataLoader): The dataloader to hold the testing data.
        model (nn.Module): The model to optimize.
        loss_fn (nn.{loss_fn}): The loss function to use during training and testing.
        optimizer (torch.optim.{optimizer}): The optimizer to use during training.
        device (str): The device to train and test the model on.
        early_stop (int, optional): If early stop is an int, model will stop optimizing if test_average_loss
                                    does not decrease in that amount of epochs. Defaults to no early stopping.
        save (bool, optional): Whether or not to save the best performing model as the optimization goes. Defaults to True.
        model_path (str, optional): Where to save model to if save is True. Defaults to "models/model.pth".
        test_type (str, optional): The type of testing framework to use. Choices are "regular" or "unet".
        quiet (bool, optional): Whether or not to supress output during model optimization. Defaults to True.
    """
    start_time = time.time()
    current_stop = early_stop if early_stop is not None else -1
    min_test_average_loss = None
    previous_test_average_loss = -1
    for t in range(epochs):
        if not quiet:
            print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        if test_type == "unet":
            test_accuracy, test_average_loss = unet_test(
                test_dataloader, model, loss_fn, device
            )
        else:
            test_accuracy, test_average_loss = test(
                test_dataloader, model, loss_fn, device
            )
        if min_test_average_loss is None:
            min_test_average_loss = test_average_loss
        if not quiet:
            print_epoch_results(train_loss, test_accuracy, test_average_loss)
        if save and test_average_loss <= min_test_average_loss:
            torch.save(model.state_dict(), model_path)
            if not quiet:
                print(f"Saved Best PyTorch Model State to {model_path}")
            min_test_average_loss = test_average_loss
        if current_stop is not None and test_average_loss >= previous_test_average_loss:
            current_stop -= 1
            if current_stop == 0:
                if not quiet:
                    print(
                        "Early stop has been triggered due to test loss not improving"
                    )
                break
        else:
            current_stop = early_stop
        previous_test_average_loss = test_average_loss
        if not quiet:
            print("-------------------------------\n")
        torch.cuda.empty_cache()
    if not quiet:
        elapsed_time = time.time() - start_time
        print(f"Finished in {round(elapsed_time, 2)}s")


def train(dataloader, model, loss_fn, optimizer, device) -> float:
    """
    Trains a given model on a dataset using a loss function, optimizer, and a device.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader holding the dataset.
        model (nn.Module): The model to test on.
        loss_fn (nn.{loss_fn}): The loss function to test on.
        optimizer (torch.optim.{optimizer}): The optimizing algorithm to use.
        device (str): The device to test on.

    Returns:
        float: The minimum loss that was encountered during training.
    """
    model.train()
    final_loss = -1
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, _ = loss.item(), (batch + 1) * len(X)
            final_loss = loss
    return final_loss


def unet_test(dataloader, model, loss_fn, device) -> tuple[float, float]:
    """
    Test a model using a given dataset, loss function, and device to test on.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader holding the dataset.
        model (nn.Module): The model to test on.
        loss_fn (nn.{loss_fn}): The loss function to test on.
        device (str): The device to test on.

    Returns:
        tuple[float, float]: The accuracy as a percentage and the test loss.
    """
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    total_pixels = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_classes = pred.argmax(dim=1)
            y = y.squeeze(1)

            correct_batch = (pred_classes == y).type(torch.float).sum().item()
            # Total pixels in the current batch
            batch_pixels = y.numel()

            # Accumulate correct predictions and total pixels
            correct += correct_batch
            total_pixels += batch_pixels
    test_loss /= num_batches
    accuracy = (float(correct) / float(total_pixels)) * 100
    return (accuracy, test_loss)


def test(dataloader, model, loss_fn, device, quiet=True) -> tuple[float, float]:
    """
    Test a model using a given dataset, loss function, and device to test on.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader holding the dataset.
        model (nn.Module): The model to test on.
        loss_fn (nn.{loss_fn}): The loss function to test on.
        device (str): The device to test on.

    Returns:
        tuple[float, float]: The accuracy as a percentage and the test loss.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= float(size)
    return (correct * 100, test_loss)


def print_epoch_results(
    train_loss, test_accuracy, test_average_loss, precision=5
) -> None:
    """
    Prints the results from an epoch.

    Args:
        train_loss (float): The resulting training loss.
        test_accuracy (float): The accuracy of the testing.
        test_average_loss (float): The average loss of the testing.
        precision (int): Amount of decimal places to show in results. Defaults to 2.
    """
    print(f"Training loss: {round(train_loss, precision)}")
    print(f"Test accuracy: {round(test_accuracy, precision)}%")
    print(f"Test average loss: {round(test_average_loss, precision)}")
