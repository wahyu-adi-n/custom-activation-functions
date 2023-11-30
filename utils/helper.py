import torch 
import torch.nn as nn
import torch.nn.functional as F

def calculate_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, dim=1)                   # extract the prediction
    num_correct    = torch.sum(predictions == labels).item()     # count how many correct predictions (.item() to obtain a python-number)
    perc_correct   = torch.tensor(num_correct/len(predictions))

    return perc_correct

def training_step(model, loader, loss_function, optimizer, device):

    # Training-mode
    model.train()

    # For every epoch initialize loss and number of correct predictions
    epoch_loss = 0
    epoch_correct = 0

    #---------------------------------------------------------- Batch-loop ---------#
    for images, labels in iter(loader):                                             #
                                                                                    #
        # Load images and labels to 'device'                                        #
        images, labels = images.to(device), labels.to(device)                       #
                                                                                    #
        # Initialize the gradient                                                   #
        optimizer.zero_grad()                                                       #
                                                                                    #
        #-------------------------------------------------------- Training ----#    #
        with torch.set_grad_enabled(True):                                     #    #
                                                                               #    #
            # Output from the model (from the forward pass)                    #    #
            output = model(images)                                             #    #
                                                                               #    #
            # Calculate the loss_function for the current batch                #    #
            loss = loss_function(output, labels)                               #    #
                                                                               #    #
            # Perform the backpropagation (backpropagate the error)            #    #
            loss.backward()                                                    #    #
                                                                               #    #
            # Gradient descent step to update parameters (weights/biases)      #    #
            optimizer.step()                                                   #    #
                                                                               #    #
            # Extract predictions                                              #    #
            _, predictions = torch.max(output, dim=1)                          #    #
        #----------------------------------------------------------------------#    #
                                                                                    #
        # Update loss (+= loss * num_images_in_the_batch)                           #
        # (.item(): returns the value of the tensor as a standard number)           #
        epoch_loss += loss.item()*images.size(0)                                    #
                                                                                    #
        # Update correct                                                            #
        epoch_correct += torch.sum(predictions == labels)                           #
    #-------------------------------------------------------------------------------#

    # Get the right epoch loss (element_loss / n_element)
    epoch_loss = epoch_loss / len(loader.dataset)

    # Accuracy of the current batch (correct / n_samples)
    accuracy = epoch_correct.double() / len(loader.dataset)

    return epoch_loss, accuracy

def evaluate_model(model, loader, loss_function, device):

    # Evaluation-mode
    model.eval()

    # For every epoch initialize loss and number of correct predictions
    epoch_loss = 0
    epoch_correct = 0

    #---------------------------------------------------------- Batch-loop ---------#
    for images, labels in iter(loader):                                             #
                                                                                    #
        # Load images and labels to 'device'                                        #
        images, labels = images.to(device), labels.to(device)                       #
                                                                                    #
        #------------------------------------------------------ Evaluation ----#    #
        with torch.set_grad_enabled(False):                                    #    #
                                                                               #    #
            # Output from the model (from the forward pass)                    #    #
            output = model(images)                                             #    #
                                                                               #    #
            # Calculate the loss_function for the current batch                #    #
            loss = loss_function(output, labels)                               #    #
                                                                               #    #
            # Extract predictions                                              #    #
            _, predictions = torch.max(output, dim=1)                          #    #
        #----------------------------------------------------------------------#    #
                                                                                    #
        # Update loss (+= loss * num_images_in_the_batch)                           #
        # (.item(): returns the value of the tensor as a standard number)           #
        epoch_loss += loss.item()*images.size(0)                                    #
                                                                                    #
        # Update correct                                                            #
        epoch_correct += torch.sum(predictions == labels)                           #
    #-------------------------------------------------------------------------------#

    # Get the right epoch loss (element_loss / n_element)
    epoch_loss = epoch_loss / len(loader.dataset)

    # Accuracy of the current batch (correct / n_samples)
    accuracy = epoch_correct.double() / len(loader.dataset)

    return epoch_loss, accuracy

# Replace ReLU with Other AFs in the model
def replace_afs(module, func):
    for child_name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, child_name, func)
        else:
            replace_afs(child, func)

def get_probs_and_preds(model, loader, device):

    model.eval()

    images_savings = []
    labels_savings = []
    probs_savings  = []
    preds_savings  = []

    #---------------------------------------------------------- Batch-loop ---------#
    for images, labels in iter(loader):                                             #
        images, labels = images.to(device), labels.to(device)                       #
        #------------------------------------------------------ Evaluation ----#    #
        with torch.set_grad_enabled(False):                                    #    #
            output = model(images)                                             #    #
            output = F.softmax(output, dim=1)  # Adjust the dimension according to your tensor structure                                       #    #
            probabilities, predictions = torch.max(output, dim=1)              #    #
        #----------------------------------------------------------------------#    #
        images_savings.append(images.cpu())                                         #
        labels_savings += labels.tolist()                                           #
        probs_savings  += probabilities.tolist()                                    #
        preds_savings  += predictions.tolist()                                      #
    #-------------------------------------------------------------------------------#

    # Accuracy
    correct_elements = 0
    for i in range(len(labels_savings)):
        if labels_savings[i] == preds_savings[i]:
            correct_elements += 1
    accuracy = correct_elements/len(labels_savings)

    return images_savings, labels_savings, probs_savings, preds_savings, accuracy