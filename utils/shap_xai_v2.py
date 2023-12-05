import torch 
import torch.nn as nn
import custom_afs
import os 
import shap
import numpy as np
import glob
from torchvision import models, transforms
from helper import replace_afs
from torch.utils.data import DataLoader
from PIL import Image

# Load saved model 
model_weights = "Dslope_1.25_DenseNet201.pt"
best_model_path = f"assets/weights/custom_layer_original/densenet/{model_weights}"

# Load Dataset
data_path = 'data/'
train_data = torch.load(os.path.join(data_path,'clahe_train_data.pkl'))
val_data = torch.load(os.path.join(data_path, 'clahe_val_data.pkl'))
test_data = torch.load(os.path.join(data_path, 'clahe_test_data.pkl'))

test_data_transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                            ])

model = models.densenet201()
model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 64),
                                nn.Linear(64, 32),
                                nn.Dropout(0.2),
                                nn.Linear(32, 2))
    
model.load_state_dict(torch.load(best_model_path))

replace_afs(module = model, func = custom_afs.DSlopeReLU(1.25))

# Use CPU
device = torch.device('cuda')
model = model.to(device)

#Load 100 images for background
shap_loader = DataLoader(train_data, batch_size=100, shuffle=True)
background, _ = next(iter(shap_loader))
background = background.to(device)

#Create SHAP explainer 
explainer = shap.DeepExplainer(model, background)

# Load test images
test_loader = DataLoader(test_data, batch_size=5, shuffle=False)
test_input, _ = next(iter(test_loader))
test_input = test_input.to(device)

# Get SHAP values
shap_values = explainer.shap_values(test_input)

# Reshape shap values and images for plotting
shap_numpy = list(np.array(shap_values).transpose(0,1,3,4,2))
test_numpy = np.array([np.array(img) for img in test_loader])

shap.image_plot(shap_numpy, test_numpy,show=False)