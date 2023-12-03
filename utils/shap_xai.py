from torchvision import models
import os
import numpy as np
import torch
import torch.nn as nn
import shap 

def shap_explainable_ai(model, data_loader, device):
    batch = next(iter(data_loader))
    images, _ = batch

    background = images[:100].to(device)
    test_images = images[100:105].to(device)
    
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
    shap.image_plot(shap_numpy, -test_numpy)
    
if __name__ == '__main__':
    # 1. Load model to make predictions
    best_model_path = "assets/weights/custom_layer_original/densenet/SmallNeg_0.3_DenseNet201.pt"
    model = models.densenet201()
    model.classifier = nn.Sequential(
                                  nn.Linear(model.classifier.in_features, 64),
                                  nn.Linear(64, 32),
                                  nn.Dropout(0.2),
                                  nn.Linear(32, 2)
                             )
    
    model.load_state_dict(torch.load(best_model_path))
    model.to("cpu")

    # 2. Load dataloader and feed to shap_xai func to see interpretable & explainable of AI
    data_path = 'data/'
    test_loader = torch.load(os.path.join(data_path, 'test_loader.pkl'))
    shap_explainable_ai(model, test_loader, "cpu")