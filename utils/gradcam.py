from torchvision import models
from helper import replace_afs
from PIL import Image

import torch 
import torch.nn as nn
import custom_afs
import os

test_loader = torch.load(os.path.join('data/', 'test_loader.pkl'))

# defines two global scope variables to store our gradients and activations
gradients = None
activations = None
device = "cuda"

def backward_hook(module, grad_input, grad_output):
    global gradients # refers to the variable in the global scope
    print('Backward hook running...')
    gradients = grad_output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    print(f'Gradients size: {gradients[0].size()}') 
    # We need the 0 index because the tensor containing the gradients comes
    # inside a one element tuple.

def forward_hook(module, args, output):
    global activations # refers to the variable in the global scope
    print('Forward hook running...')
    activations = output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    print(f'Activations size: {activations.size()}')


model_weights = "Dslope_1.25_DenseNet201.pt"
best_model_path = f"assets/weights/custom_layer_original/densenet/{model_weights}"

model = models.densenet201()
model.classifier = nn.Sequential(
                                  nn.Linear(model.classifier.in_features, 64),
                                  nn.Linear(64, 32),
                                  nn.Dropout(0.2),
                                  nn.Linear(32, 2)
                             )
    
model.load_state_dict(torch.load(best_model_path))
model.to("cpu")


replace_afs(module = model, func = custom_afs.DSlopeReLU(1.25))

backward_hook = model.features.register_full_backward_hook(backward_hook, prepend=False)
forward_hook = model.features.register_forward_hook(forward_hook, prepend=False)
model.eval()
for images, labels in iter(test_loader):
    # images, labels = images, labelsdevice
    
    output = model(images).backward()