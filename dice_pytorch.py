import os
import time
import pickle
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
import matplotlib
import matplotlib.pyplot as plt
import DiceDataset

matplotlib.use('Agg')
device = ("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Neural network used for dice
class DiceCNN(nn.Module):
    def __init__(self):
        super(DiceCNN,self).__init__()
        # output 6
        k1, k2, k3 = 3, 3, 3 # kernal size
        s1, s2, s3 = 1,1,1 # stride size
        self.n_features = 16*58*58
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=k1,stride=s1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2,2)),
            #nn.MaxPool2d(50),
            nn.Conv2d(32, 32, kernel_size=k2,stride=s2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 16, kernel_size=k2,stride=s2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2,2))
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_features, 200),
            nn.ReLU(),
            nn.Linear(200,6),
            nn.Sigmoid()
        )
    def forward(self,x):
#        print("x", x.size())
        x = self.network(x)
        #x = x.view(x.shape[0],-1)
#        print(x.view(x.shape[0],-1).size())
#        print("x", x.size())
        return self.out(x)

# TODO Download data
def downloadData():
    return

# TODO move training from main to trainModel
def trainModel():
    return

def save_model(model, save_path):
    torch.save(model.state_dict, save_path)
    print("Saving new best model")

def load_model(model,load_path):
    model.load_state_dict(torch.load(load_path))
    return model

def plot_results(train_loss, val_loss):
    figure = plt.figure(1)
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.plot(train_loss)
    plt.plot(val_loss)
    figure.savefig("results/loss.png")

if __name__ == "__main__":
    if not os.path.isdir("results"):
        print("Creating results folder")
        os.mkdir("results/")

    if not os.path.isdir("dice/"):
        print("Warning, dice dataset(\"dice/\")cannot be found. \n ")
        userresponse = input("Would you like to download? [Y,n]")
        if(userresponse == "Y" or userresponse == "y"):
            print("Download function not implemented yet. Please download yourself. https://www.kaggle.com/ucffool/dice-d4-d6-d8-d10-d12-d20-images/notebooks")
    else:
        print("Found dice dataset. Now preparing...")

    transform = transforms.Compose([transforms.Resize((100,100)),
                                    transforms.ToTensor()])
    # Load dataset
    traindataset = DiceDataset.DiceDataset(root="dice/train",transform=None, preprocess=True)
    train_set, validation_set = torch.utils.data.random_split(traindataset,[9999, 4285])
    validdataset = DiceDataset.DiceDataset(root="dice/valid",transform=transform, preprocess=True)

    # Hyperparameters
    epochs = 100
    learning_rate = 7e-5
    train_CNN = False
    batch_size = 32
    shuffle = True
    pin_memory = True
    num_workers = 10

    # Saving and plotting
    save_iteration = 10
    plot_iteration = 5

    # Create dataloader
    train_loader = DataLoader(dataset=train_set, shuffle = shuffle, batch_size = batch_size, num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(dataset=validation_set, shuffle = shuffle, batch_size = batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # Define model
    model = DiceCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    val_loss_array = []
    train_loss_array = []
    # Train model
    for epoch in range(epochs):
            training_loss = 0.0
            val_loss = 0.0
            val_acc = 0
            correct_preds = 0
            validation = 0.0
            total = 0
            model.train()
            print("Epoch ",epoch)
            for i,data in enumerate(train_loader,0):
                imgs, labels = data
                if torch.cuda.is_available():
                    #print("imgs",imgs.shape)
                    #print("labels",labels.shape)
                    imgs = imgs.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.int64)
                outputs = model(imgs.float())
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

            # Run on validation set
            with torch.no_grad():
                model.eval()
                for i,data in enumerate(validation_loader,0):
                    imgs, labels = data
                    if torch.cuda.is_available():
                        imgs = imgs.to(device, dtype=torch.float)
                        labels = labels.to(device, dtype=torch.int64)
                    outputs = model(imgs)

                    val_loss = criterion(outputs,labels)
                    _, index = torch.max(outputs,1)
                    total += labels.size(0)
                    correct_preds += (index == labels).sum().item()

                    validation += val_loss.item()

            # Record results after evalutation
            train_loss_array.append(training_loss)
            val_loss_array.append(validation)
            val_acc = 100 * (correct_preds / total)
            # Saving models
            if(best_acc < val_acc):
                best_acc = val_acc
                save_model(model,"results/bestmodel.pth")
            if(epoch % save_iteration == 0):
                save_model(model, "results/lastsavedmodel.pth")
            # Plot
            if( epoch % plot_iteration == 0):
                plot_results(train_loss_array,val_loss_array)

            print("val_acc %5.2f training_loss % 5.2f val_loss % 5.2f, best_acc % 5.2f" % (val_acc,training_loss, val_loss, best_acc))

