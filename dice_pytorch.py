from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import numpy as np
# Project imports
import DiceDataset

class DiceCNN(nn.Module):
    def __init__(self):
        super(DiceCNN,self).__init__()

    def forward(self,x):
        return x


if __name__ == "__main__":
    train_dataset = DiceDataset(root="dices/train",transform=True)
    test_dataset = DiceDataset(root="dices/test",transform=True)

