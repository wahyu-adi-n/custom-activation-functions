import torch
import os

def class_counting():
    # Set counters
    n_samples_class1_train = len(os.listdir('chest_xray/train/NORMAL'))
    n_samples_class2_train = len(os.listdir('chest_xray/train/PNEUMONIA'))

    # Define two dictionaries
    class_count = {0: n_samples_class1_train, 
                   1: n_samples_class2_train}
    
    return class_count

def weighted_class(device):
    class_count = class_counting()
    # Class weights values
    samples_0 = class_count[0]
    samples_1 = class_count[1]
    tot_samples = samples_0 + samples_1

    weight_0 = 1 - samples_0/tot_samples
    weight_1 = 1 - weight_0  # equivalent to = 1 - samples_1/tot_samples

    # Class weights tensor
    class_weights = [weight_0, weight_1]
    class_weights = torch.cuda.FloatTensor(class_weights) if device == "cuda" else torch.FloatTensor
    print(class_weights)
    
    return class_weights

if __name__ == "__main__":
    print(f"Class weighting: {weighted_class(device='cuda')}")