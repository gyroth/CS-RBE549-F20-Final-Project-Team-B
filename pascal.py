from __future__ import print_function, division
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        dice_type = re.search("/([A-z]*([0-9]+))+/(?!.*\1)",img_path).group().replace('/','') # gets type of dice
        img = preprocessImage(img_path) if self.preprocess else cv2.imread(img_path)
        return (dice_type , self.transform(img) if self.transform else img)

    # Insert any image preprocessing here
    def preprocessImage(image_path):
        img = cv2.imread(image_path)
        canny = cv2.Canny(img, 30, 150)
        return img

# Rescale image
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        dice_type , img = sample
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img_resized = transform.resize(image, (new_h, new_w))
        return (dice_type, img_resized)

if __name__ == "__main__":
    train_dataset = DiceDataset(root="dices/train",transform=True)
    test_dataset = DiceDataset(root="dices/test",transform=True)
