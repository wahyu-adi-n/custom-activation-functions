from torch import nn
from torchvision import models
from config.config import *
from utils.data_preparation import load_data_loader_preparation
from utils.helper import *
from model.small_nn import NeuralNetwork
from model.resnet import ResidualBlock, ResNet
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import torch
import csv
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Dataset, data loaders preparation
train_loader, val_loader, test_loader = load_data_loader_preparation()

for X, y in train_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Creating the model
# Get cpu or gpu device for training.
print(f"Device Type: {device} device.")
print(f"Epochs: {epochs} epochs.")
print(f"Num Classes: {num_classes}")

with open("assets/logs/densenet201_results.csv", mode="w") as csv_file:
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(["Activation Function", "Epoch", "Training Accuracy", "Test Accuracy", "Training Loss", "Test Loss", "Time(s)", "Best Model(?)"])

    # for each activation function
    for text, func in afs_dict.items():
        
        # Define model
        # Using pre-trained models
        
        # 1. DenseNet201
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
            
        for parameter in model.parameters():
            parameter.requires_grad = False
        
        model.classifier = nn.Linear(1920, num_classes)
        
        # 2. ResNet152v2
        # model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
            
        # for parameter in model.parameters():
        #     parameter.requires_grad = False
        
        # model.fc = nn.Linear(2048, num_classes)
        
        # 3. VGG19
        # model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            
        # for parameter in model.parameters():
        #     parameter.requires_grad = False
        
        # model.classifier[6] = nn.Linear(4096, num_classes)
    
        # Add custom layers (future works)
        # model.classifier = nn.Sequential(
        #                           nn.Linear(model.classifier.in_features, 64),
        #                           nn.Linear(64, 32),
        #                           nn.Dropout(0.2),
        #                           nn.Linear(32, num_classes)
        #                      )
            
        model.to(device)

        # model = NeuralNetwork().to(device)
        # model = ResNet(ResidualBlock, [3,1,2,4]).to(device)
        # model = ResNet(ResidualBlock, [3,4,6,3]).to(device)

        # print("Before:\n", model)

        # Replace afs in hidden layers
        replace_afs(module = model, func = func)  

        # print("\nAfter replace AFs:\n", model)

        # Optimizing the model parameters
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        
        # Monitor 'val_loss'
        best_val_loss = float('inf')

        # For the records
        train_loss_savings = []
        train_acc_savings  = []
        val_loss_savings   = []
        val_acc_savings    = []

        # Saving the model
        best_model = copy.deepcopy(model.state_dict())
        
        # Counter for early stopping
        early_stopping_counter = 0
        is_best = False

        # ======================================
        #   TRAINING STEP
        # ======================================
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            # Init Time
            loop_start_time = time.perf_counter()
            
            # Training step
            train_loss, train_acc = training_step(model, 
                                                  train_loader, 
                                                  loss_function, 
                                                  optimizer, 
                                                  device)
            train_loss_savings.append(train_loss)
            train_acc_savings.append(train_acc.item())

            # Evaluation step
            val_loss, val_acc = evaluate_model(model, 
                                               val_loader, 
                                               loss_function, 
                                               device)
            val_loss_savings.append(val_loss)
            val_acc_savings.append(val_acc.item())

            # Print results
            print(f'Epoch: {epoch+1} / {epochs} - train_loss: {train_loss:.4f} - train_accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}')

            # If the val_loss improved, save the model
            if val_loss < best_val_loss:
                print(f'Epoch: {epoch+1:02}/{epochs} - val_loss improved from {best_val_loss:.4f} to {val_loss:.4f}, new model saved')
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0  # reset the counter
                is_best = True
            else:
                print(f'Epoch: {epoch+1:02}/{epochs} - val_loss did not improve')
                early_stopping_counter += 1
                is_best = False
                # Check if early stopping criteria are met
                if early_stopping_counter >= patience:
                    print(f'Early stopping! No improvement for {patience} consecutive epochs.')
                    break  # Stop training
                    
            # Update scheduler (learning rate adapter)
            scheduler.step()
            
            # Write results to csv [act_func, epoch, train acc, test acc, train loss, test loss]
            # Last element in accuracy and lost list should be results for the epoch that it has just done
            loop_end_time = time.perf_counter()
            csv_file_writer.writerow([text, epoch+1, 
                                      train_acc_savings[-1], 
                                      val_acc_savings[-1], 
                                      train_loss_savings[-1], 
                                      val_loss_savings[-1], 
                                      loop_end_time - loop_start_time,
                                      is_best])
        print("Training Complete!")

        # Saving the model
        # save the model with afs in the name as well iteration
        path_best_model = f"assets/weights/"
        torch.save(model.state_dict(), path_best_model + f"{text}_{model_name}.pt")
        print(f"Saved PyTorch Model State to {text}-{model_name}.pt\n")
            
        model.load_state_dict(torch.load(path_best_model + f"{text}_{model_name}.pt"))

        # Creating plot
        epoch = range(1, len(val_loss_savings) +1)

        # Plottinf Model Accuracy Curve
        plt.plot(epoch, train_acc_savings, label="train")
        plt.plot(epoch, val_acc_savings, label="val")
        plt.title(f"{text}-{model_name} Model Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(f"assets/acc_plots/{text}_{model_name}_Model_Accuracy.png")
        plt.figure()
        plt.clf()

        # Plotting Model Loss Curve
        plt.plot(epoch, train_loss_savings, label="train")
        plt.plot(epoch, val_loss_savings, label="val")
        plt.title(f"{text}-{model_name} Model Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(f"assets/loss_plots/{text}_{model_name}_Model_Loss.png")
        plt.figure()
        plt.clf()
        
        # Evaluate the model on test data
        images, labels, probs, preds, accuracy = get_probs_and_preds(model, test_loader, device)

        # Calculate additional metrics
        report = classification_report(labels, preds)
        
        # Specify the file path where you want to save the report
        file_path = f'assets/classification_report/Classification_Report_{text}_{model_name}.txt'

        # Write the report to the file
        with open(file_path, 'w') as file:
            file.write(f'Accuracy on test dataloader: {accuracy:.4f}\n')
            file.write(report)
            
        print(f'Classification report saved to assets/classification_report')

        # Print results
        print(f'Accuracy on test dataloader: {accuracy:.4f}')
        print(report)
        
        # Build the confusion matrix
        cm = confusion_matrix(labels, preds)

        # Normalize the confusion matrix
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Ticklables
        ticklabels = class_index.values()

        # Plot
        fig, ax = plt.subplots(figsize=(15,5))

        # Confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='.3g', 
                    xticklabels=ticklabels, 
                    yticklabels=ticklabels, 
                    cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Normalized confusion matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(cmn, annot=True, fmt='.3f', 
                    xticklabels=ticklabels, 
                    yticklabels=ticklabels, 
                    cmap=plt.cm.Blues);
        plt.title('Normalized Confusion Matrix');
        plt.xlabel('Predicted');
        plt.ylabel('Actual'); 

        plt.subplots_adjust(wspace=.3)
        
        # Save the figure
        plt.savefig(f'assets/confusion_matrix/Confusion_Matrix_{text}_{model_name}.png')
        plt.figure()
        plt.clf()

        print(f'Classification report saved to assets/confusion_matrix')