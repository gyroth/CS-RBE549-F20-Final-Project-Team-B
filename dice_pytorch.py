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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

device = ("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
num_dices = 6

class DiceCNN(nn.Module):
    def __init__(self):
        super(DiceCNN,self).__init__()
        # output 6
        k1, k2, k3 = 8, 3, 3
        s1, s2, s3 = 2,1,1
        self.n_features = 1*233*233
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=k1,stride=s1),
            #nn.MaxPool2d(50),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=k2,stride=s2),
            #nn.MaxPool2d(50),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=k2,stride=s2),

        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_features, 200),
            nn.ReLU(),
            nn.Linear(200,6),
            nn.Sigmoid()
        )
    def forward(self,x):
        #print("x", x.size())
        x = self.network(x)
        #x = x.view(x.shape[0],-1)
        #print("x", x.size())
        return self.out(x)

def save_best_model(model, save_path):
    torch.save(model.state_dict, save_path)
    print("Saving new best model")

def load_model(model,load_path):
    model.load_state_dict(torch.load(load_path))
    return model

# TODO
def plot_results(train_loss, val_loss):
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.savefig("results/loss.png")

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((100,100)),
                                    transforms.ToTensor()])
    # Load dataset
    traindataset = DiceDataset.DiceDataset(root="dice/train",transform=None, preprocess=True)
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
            total =0
            model.train()
            print("Epoch % 2d, training_loss % 5.2f val_loss % 5.2f, best_acc % 5.2f" % (epoch,training_loss, val_loss, best_acc))
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
                print('Validation Accuracy is: {:.2f}%'.format(val_acc))
                # Saving models
                if(best_acc > val_acc):
                    save_best_model(model,"results/bestmodel.pth")
                if(epoch % save_iteration == 0):
                    save_best_model(model, "results/lastsavedmodel.pth")
                # Plot
                if( epoch % plot_iteration == 0):
                    plot_results(train_loss_array,val_loss_array)


