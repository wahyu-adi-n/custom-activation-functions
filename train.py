from torch import nn
from torchvision import models
from config.config import afs_dict, device, epochs
from utils.data_preparation import data_prep, load_data_loader_preparation
from utils.engine import train, evaluate
from utils.helper import replace_afs

import torch
import csv
import time

# Dataset, data loaders preparation
train_loader, val_loader, test_loader = load_data_loader_preparation()

for X, y in train_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Creating the model
# Get cpu or gpu device for training.
print(f"Using {device} device")

with open("densenet_121_results.csv", mode="w") as csv_file:
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(["Activation Function", "Epoch", "Training Accuracy", "Test Accuracy", "Training Loss", "Test Loss", "Time(s)"])

    # for each activation function
    for text, func in afs_dict.items():

        #test each function 1 times in order to caluclate statistics
        for i in range(1, 2):

            # Define model

            # Using pre-trained models
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            model.to(device)

            # model = NeuralNetwork().to(device)
            # model = ResNet(ResidualBlock, [3,1,2,4]).to(device)
            # model = ResNet(ResidualBlock, [3,4,6,3]).to(device)
            # print(model)
            
            # Replace afs in hidden layers
            replace_afs(module=model, func=func)  
            
            # Optimizing the model parameters
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            train_accuracy_list =[]
            test_accuracy_list =[]
            train_loss_list =[]
            test_loss_list =[]

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
            # save the model with act func in the name as well iteration
            # torch.save(model.state_dict(), f"{text}_{i}.pth")
            # print(f"Saved PyTorch Model State to {text}_{i}.pth")

            # Creating plot
            # epochs = range(1, len(test_accuracy_list) +1)
            #
            # plt.plot(epochs, train_accuracy_list, "bo", label="Train acc")
            # # plt.plot(epochs, val_acc, "b", label="Validation acc")
            # plt.title(f"{text}_ResNetTrain accuracy_{i}")
            # plt.legend()
            # plt.savefig(f"{text}_ResNetTrain accuracy_{i}.png")
            # # plt.figure()
            #
            # plt.clf()
            #
            # plt.plot(epochs, test_accuracy_list, "bo", label="Test acc")
            # # plt.plot(epochs, val_acc, "b", label="Validation acc")
            # plt.title(f"{text}_ResNetTest accuracy_{i}")
            # plt.legend()
            # plt.savefig(f"{text}_ResNetTest accuracy_{i}.png")
            #
            # plt.clf()
        # plt.figure()

# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()
# plt.savefig("loss.png")
# plt.show()