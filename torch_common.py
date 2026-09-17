import os

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset


cpu, gpu = torch.device("cpu"), torch.device("cuda")

MODEL_PATH = r"checkpoints\\"
# MODEL_PATH = r""

def makeModelPath():
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

def loadModel(fileName = "latest.pt"):
    fileName=MODEL_PATH+fileName
    if os.path.exists(fileName):
        with open(fileName, "rb") as f: 
            # return torch.load(f, weights_only=False)
            return torch.load(f, weights_only=False)
    else:
        raise Exception(f"no model called {fileName}")

def saveModel(model, fileName = "latest.pt"):
    makeModelPath()
    fileName=MODEL_PATH+fileName
    with open(fileName, "wb") as f:
        torch.save(model, f)

from torch.utils.data import random_split

def get_train_test_split(dataset, split, bsize = 32, doShuffle = True):
    if split is None:
        return DataLoader(dataset, bsize, doShuffle)

    size = len(dataset)
    testSize = int(split * size)
    trainSize = size - testSize

    return [
        # the dataLOADER created from the dataSET returned by .. 
        DataLoader(dataSet, bsize, doShuffle) 
            for dataSet in 

                # .. randomly splitting the whole dataset
                torch.utils.data.random_split(
                                    dataset, 
                                    [int(trainSize), int(testSize)])
    ]
    # return random_split(dataset, [1.0 - split, split])