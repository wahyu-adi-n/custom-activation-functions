from torchvision import transforms, datasets
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from collections import Counter

import torch
import numpy as np

def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set this to False for reproducibility

def data_prep():
    set_random_seeds()  # Set random seeds before data preparation
        
    train_data_transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.RandomGrayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ])

    test_data_transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ])

    # Assosciate the dataset and the transformations with ImageFolder
    train_data = datasets.ImageFolder('chest_xray/train', transform = train_data_transform)
    val_data = datasets.ImageFolder('chest_xray/val', transform = test_data_transform)
    test_data = datasets.ImageFolder('chest_xray/test', transform = test_data_transform)
    
    return train_data, val_data, test_data


def load_data_loader_preparation(batch_size: int = 64):
    set_random_seeds()  # Set random seeds before data preparation
        
    train_data, val_data, test_data = data_prep()
    # Create data loaders.
    train_loader = DataLoader(train_data, 
                              batch_size = batch_size, 
                              shuffle = True)
    val_loader   = DataLoader(val_data, 
                              batch_size = batch_size,
                              shuffle =  False)
    test_loader   = DataLoader(test_data,
                               batch_size = batch_size, 
                               shuffle =  False)
    
    return train_loader, val_loader, test_loader                                           

def load_data_loader_preparation_smote(batch_size: int = 64):
    set_random_seeds()  # Set random seeds before data preparation
        
    train_data, val_data, test_data = data_prep()
    
    # Apply SMOTE to all data
    X_train = np.array([item[0].numpy().flatten() for item in train_data])
    y_train = np.array([item[1] for item in train_data])
    
    X_val = np.array([item[0].numpy() for item in val_data])
    y_val = np.array([item[1] for item in val_data])
    
    X_test = np.array([item[0].numpy() for item in test_data])
    y_test = np.array([item[1] for item in test_data])
    
    # Reshape the image arrays to 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    
    X_resampled_train, y_resampled_train = smote.fit_resample(X_train_flat, y_train)
    
    X_resampled_val, y_resampled_val = smote.fit_resample(X_val_flat, y_val)
    
    X_resampled_test, y_resampled_test = smote.fit_resample(X_test_flat, y_test)
    
    counter_resampled = Counter(y_resampled_train)

    # Print the count for each class
    for class_label, count in counter_resampled.items():
        print(f"Class {class_label}: {count} instances")

    # Convert NumPy arrays to PyTorch tensors
    X_resampled_train = torch.tensor(X_resampled_train, dtype=torch.float32)
    y_resampled_train = torch.tensor(y_resampled_train, dtype=torch.long)
    
    X_resampled_val = torch.tensor(X_resampled_val, dtype=torch.float32)
    y_resampled_val = torch.tensor(y_resampled_val, dtype=torch.long)
    
    X_resampled_test = torch.tensor(X_resampled_test, dtype=torch.float32)
    y_resampled_test = torch.tensor(y_resampled_test, dtype=torch.long)
    
    # Create PyTorch datasets and dataloaders
    train_data_smote = TensorDataset(X_resampled_train, y_resampled_train)
    val_data_smote =  TensorDataset(X_resampled_val, y_resampled_val)
    test_data_smote = TensorDataset(X_resampled_test, y_resampled_test)

    # Create data loaders.
    train_loader = DataLoader(train_data_smote, 
                              batch_size = batch_size, 
                              shuffle = True)
    val_loader   = DataLoader(val_data_smote, 
                              batch_size = batch_size, 
                              shuffle =  False)
    test_loader   = DataLoader(test_data_smote, 
                               batch_size = batch_size, 
                               shuffle =  False)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    
    saved_data_path = 'data/'

    if not os.path.exists(saved_data_path):
        os.makedirs(saved_data_path, exist_ok=True)

    train_data, val_data, test_data =  data_prep()

    torch.save(train_data, os.path.join(saved_data_path, 'train_data.pkl'))
    torch.save(val_data, os.path.join(saved_data_path, 'val_data.pkl'))
    torch.save(test_data, os.path.join(saved_data_path, 'test_data.pkl'))
    print("[INFO] Data succesfully saved!")
    
    train_loader, val_loader, test_loader = load_data_loader_preparation()

    torch.save(train_loader, os.path.join(saved_data_path,'train_loader.pkl'))
    torch.save(val_loader, os.path.join(saved_data_path, 'val_loader.pkl'))
    torch.save(test_loader, os.path.join(saved_data_path, 'test_loader.pkl'))

    print("[INFO] Dataloader succesfully saved!")
    
    train_loader_smote, val_loader_smote, test_loader_smote = load_data_loader_preparation()

    torch.save(train_loader_smote, os.path.join(saved_data_path,'train_loader_smote.pkl'))
    torch.save(val_loader_smote, os.path.join(saved_data_path, 'val_loader_smote.pkl'))
    torch.save(test_loader_smote, os.path.join(saved_data_path, 'test_loader_smote.pkl'))

    print("[INFO] SMOTE Dataloader succesfully saved!")