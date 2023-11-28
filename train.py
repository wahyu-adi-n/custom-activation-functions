from torch import nn
from torchvision import models
from config.config import afs_dict, device, epochs
from utils.data_preparation import load_data_loader_preparation
from utils.helper import replace_afs
from model.small_nn import NeuralNetwork
from model.resnet import ResidualBlock, ResNet

import torch
import csv
import time
import matplotlib.pyplot as plt

# Creating training loop
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    train_accuracy_list.append(correct)
    train_loss_list.append(train_loss)

# Creating testing loop
def evaluate(dataloader, model, loss_fn, device):
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
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_accuracy_list.append(correct)
    test_loss_list.append(test_loss)

# Dataset, data loaders preparation
train_loader, val_loader, test_loader = load_data_loader_preparation()

for X, y in train_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Creating the model
# Get cpu or gpu device for training.
print(f"Using {device} device")

# Define model

# Using pre-trained models
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.to(device)

# model = NeuralNetwork().to(device)
# model = ResNet(ResidualBlock, [3,1,2,4]).to(device)
# model = ResNet(ResidualBlock, [3,4,6,3]).to(device)
print(model)

with open("densenet_121_results.csv", mode="w") as csv_file:
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(["Activation Function", "Epoch", "Training Accuracy", "Test Accuracy", "Training Loss", "Test Loss", "Time(s)"])

    # for each activation function
    for text, func in afs_dict.items():

        #test each function 1 times in order to caluclate statistics
        for i in range(1, 2):
            # Replace afs in hidden layers
            replace_afs(module = model, func = func)  

            # Optimizing the model parameters
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

            train_accuracy_list = []
            test_accuracy_list = []
            train_loss_list = []
            test_loss_list = []

            # Training and testing the network
            for t in range(epochs):
                loop_start_time = time.perf_counter()
                print(f"Epoch {t+1}\n-------------------------------")
                train(train_loader, model, loss_fn, optimizer, device)
                evaluate(val_loader, model, loss_fn, device)
                
                # Update scheduler (learning rate adapter)
                scheduler.step()
                
                # Write results to csv [act_func, epoch, train acc, test acc, train loss, test loss]
                # Last element in accuracy and lost list should be results for the epoch that it has just done
                loop_end_time = time.perf_counter()
                csv_file_writer.writerow([text, t+1, train_accuracy_list[-1], test_accuracy_list[-1], train_loss_list[-1], test_loss_list[-1], loop_end_time - loop_start_time])
            
            print("Training Done!")

            # Saving the model
            # save the model with afs in the name as well iteration
            torch.save(model.state_dict(), f"{text}_{i}.pth")
            print(f"Saved PyTorch Model State to {text}_{i}.pth")

            # Creating plot
            epoch = range(1, len(test_accuracy_list) +1)
                        
            plt.plot(epoch, train_accuracy_list, "b", label="Train Accuracy")
            # plt.plot(epoch, val_acc, "bo", label="Validation acc")
            plt.title(f"{text} DenseNet121 Train_Accuracy {i}")
            plt.legend()
            plt.savefig(f"assets/{text}_DenseNet_121_Train_Acc_{i}.png")
            plt.figure()
            plt.clf()

            plt.plot(epoch, test_accuracy_list, "b", label="Val Accuracy")
            # plt.plot(epoch, val_acc, "bo", label="Validation acc")
            plt.title(f"{text} DenseNet121 Val Accuracy {i}")
            plt.legend()
            plt.savefig(f"assets/{text}_DenseNet_121_Val_Acc_{i}.png")   
            plt.clf()

        plt.figure()
plt.plot(epoch, train_loss_list, "b", label="Train Loss")
plt.plot(epoch, test_loss_list, "b", label="Val Loss")
plt.title("Train and Val Loss")
plt.legend()
plt.savefig("assets/DenseNet_121_Train_Val_Loss.png")
plt.show()