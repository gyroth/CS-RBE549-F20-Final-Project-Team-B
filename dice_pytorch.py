import torchvision.transforms as transforms
import tqdm
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import numpy as np
# Project imports
import DiceDataset

device = ("cuda" if torch.cuda.is_available() else "cpu")

class DiceCNN(nn.Module):
    def __init__(self):
        super(DiceCNN,self).__init__()
        # output 6
        k1, k2, k3 = 3, 3, 3
        s1, s2, s3 = 1,1,1
        self.network = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=k1,stride=s1),
            nn.MaxPool2d(50),
            nn.ReLU(),
            #nn.Conv2d(50, 50, kernel_size=k2,stride=s2),
            #nn.MaxPool2d(50),
            #nn.ReLU(),
            nn.Linear(480*480*1, 200),
            nn.ReLU(),
            nn.Linear(200,6),
            nn.Sigmoid()
        )

    def forward(self,x):
        print("x shape",x.size())
        return self.network(x)




def check_accuracy(loader, model):
    if loader == train_loader:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on validation data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        #print("loader", loadertraindataset = DiceDataset.DiceDataset(root="dice/train",transform=transform, preprocess=True))
        for (x, y1) in loader:
            x = x.to(device = device, dtype=torch.float)
            y = y1[0].to(device = device, dtype=torch.float)

            scores = model(x)
            predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
            num_correct += (predictions == y[0]).sum()
            num_samples += predictions.size(0)
    return f"{float(num_correct)/float(num_samples)*100:.2f}"


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((100,100)),
                                    transforms.ToTensor()])
    # Load dataset
    traindataset = DiceDataset.DiceDataset(root="dice/train",transform=None, preprocess=True)
    split_ratio = 0.7
    trainlen = len(traindataset)*split_ratio
    validlen = len(traindataset)*(1-split_ratio)
    train_set, validation_set = torch.utils.data.random_split(traindataset,[9999, 4285])
    validdataset = DiceDataset.DiceDataset(root="dice/valid",transform=transform, preprocess=True)

    # Hyperparameters
    epochs = 100
    learning_rate = 0.07
    train_CNN = False
    batch_size = 32
    shuffle = True
    pin_memory = True
    num_workers = 1
    # Create dataloader
    train_loader = DataLoader(dataset=train_set, shuffle = shuffle, batch_size = batch_size, num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(dataset=validation_set, shuffle = shuffle, batch_size = batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # Define model
    model = DiceCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train model
    model.train()
    for epoch in range(epochs):
        loop  = tqdm.tqdm(train_loader, total = len(train_loader), leave = True)
        if epoch % 2 == 0:
            loop.set_postfix(val_acc = check_accuracy(validation_loader, model))
            for imgs, labels in loop:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(loss = loss.item())
