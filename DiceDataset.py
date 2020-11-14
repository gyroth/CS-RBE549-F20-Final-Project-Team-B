import torchvision.transforms as transforms
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import numpy as np
import os
import glob
import re
# Used for pytorch
# Preprocesses the images and loads them into a dataset
class DiceDataset(Dataset):

    # root example: dices/train/d6
    # transform: True to transform images using canny
    def __init__(self, root, transform=None, preprocess=True):
        self.image_paths = glob.glob(f"{root}/**/*.jpg", recursive=True)
        self.transform = transform
        self.preprocess = preprocess
        self.dice = [4,6,8,10,12,20]
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        dice_type = re.search("/([A-z]*([0-9]+))+/(?!.*\1)",img_path).group().replace('/','').replace('d','') # gets type of dice
        img = self.preprocessImage(img_path) if self.preprocess else cv2.imread(img_path)
        img2 = self.transform(torch.from_numpy(img)) if self.transform else torch.from_numpy(img)
        img3 = img2.unsqueeze(0)
        # I am not one hot encoding, I am assigning each category an index. I.e. 4 sided dice is label 1, 6 sided dice is converted to label 2, etc.
        dice_index = self.dice.index(float(dice_type))
        #print(dice_index)
        return (img3, torch.tensor(float(dice_index)))

    # Insert any image preprocessing here
    def preprocessImage(self,image_path):
        img = cv2.imread(image_path)
        img = cv2.Canny(img, 30, 150)
        return img
