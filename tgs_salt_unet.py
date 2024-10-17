import os
from pathlib import Path

import torch
from imutils import paths
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from models.unet.dataset import UNETDataset
from models.unet.unet import UNet
from utilities.training_pipeline import execute_model_optimization

if __name__ == "__main__":
    # Constants
    DATASET_PATH = Path().joinpath("data", "tgs-salt-identification-challenge", "train")
    IMAGE_DATASET_PATH = Path(DATASET_PATH).joinpath("images")
    MASK_DATASET_PATH = Path(DATASET_PATH).joinpath("masks")
    TEST_SPLIT = 0.15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False
    NUM_CHANNELS = 128
    NUM_CLASSES = 1
    NUM_LEVELS = 3
    LEARNING_RATE = 0.001
    EPOCHS = 5
    BATCH_SIZE = 32
    INPUT_IMAGE_WIDTH = 128
    INPUT_IMAGE_HEIGHT = 128
    THRESHOLD = 0.5
    BASE_OUTPUT = "saved_models"
    MODEL_PATH = Path().joinpath(BASE_OUTPUT, "unet_tgs_salt.pth")
    PLOT_PATH = Path().joinpath(BASE_OUTPUT, "plot.png")
    TEST_PATHS = Path().joinpath(BASE_OUTPUT, "test_paths.txt")

    # load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))[:10]
    maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))[:10]

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(
        imagePaths, maskPaths, test_size=TEST_SPLIT, random_state=42
    )

    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # write the testing image paths to disk so that we can use then
    # when evaluating/testing our model
    print("[INFO] saving testing image paths...")
    f = open(TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()

    # define transformations
    transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
            transforms.ToTensor(),
        ]
    )

    # create the train and test datasets
    trainDS = UNETDataset(
        imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms
    )
    testDS = UNETDataset(
        imagePaths=testImages, maskPaths=testMasks, transforms=transforms
    )
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(
        trainDS,
        shuffle=True,
        batch_size=BATCH_SIZE,
        pin_memory=PIN_MEMORY,
        num_workers=os.cpu_count(),
    )
    testLoader = DataLoader(
        testDS,
        shuffle=False,
        batch_size=BATCH_SIZE,
        pin_memory=PIN_MEMORY,
        num_workers=os.cpu_count(),
    )

    # initialize our UNet model
    unet = UNet(in_channels=3).to(DEVICE)
    # initialize loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=LEARNING_RATE)
    # calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // BATCH_SIZE
    testSteps = len(testDS) // BATCH_SIZE

    execute_model_optimization(
        EPOCHS,
        trainLoader,
        testLoader,
        unet,
        lossFunc,
        opt,
        DEVICE,
        early_stop=5,
        save=True,
        model_path=MODEL_PATH,
        test_type="unet",
        quiet=False,
    )
