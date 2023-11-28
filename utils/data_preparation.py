from torchvision import transforms, datasets
import torch

def data_prep():
    train_data_transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.RandomGrayscale(),
                                transforms.ToTensor()
                            ])

    test_data_transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                            ])

    # Assosciate the dataset and the transformations with ImageFolder
    train_data = datasets.ImageFolder('chest_xray/train', transform = train_data_transform)
    val_data = datasets.ImageFolder('chest_xray/val', transform = test_data_transform)
    test_data = datasets.ImageFolder('chest_xray/test', transform = test_data_transform)
    
    return train_data, val_data, test_data


def load_data_loader_preparation(batch_size: int = 64):
    train_data, val_data, test_data = data_prep()
    # Create data loaders.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader   = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle =  False)
    test_loader   = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle =  False)
    
    return train_loader, val_loader, test_loader                                           
