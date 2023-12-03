from utils.helper import replace_afs, get_probs_and_preds
from torchvision import models
from config.config import *

import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

model_weights = "SmallNeg_0.3_DenseNet201.pt"
best_model_path = f"assets/weights/custom_layer_original/densenet/{model_weights}"

model = models.densenet201()
model.classifier = nn.Sequential(
                                  nn.Linear(model.classifier.in_features, 64),
                                  nn.Linear(64, 32),
                                  nn.Dropout(0.2),
                                  nn.Linear(32, 2)
                             )
    
model.load_state_dict(torch.load(best_model_path))
model.to("cuda")

replace_afs(module = model, func = afs_dict['SmallNeg_0.3'])

print(model)

test_loader = torch.load(os.path.join('data/', 'test_loader.pkl'))

images, labels, probs, preds, accuracy = get_probs_and_preds(model, test_loader, "cuda")

## Visualize some images with: true label, predicted label, probability

plt.figure(figsize=(20, 26))
n_rows = 7
n_cols = 7

# Display images
for i in range(n_rows*n_cols):
  text_true = 'True: ' + class_index[labels[i]]
  text_pred = '\nPredicted: ' + class_index[preds[i]]
  text_prob = '\nwith probability: ' + str(probs[i])[:5]

  if labels[i] == preds[i]:
      text_correct = ' ✓'
  else:
      text_correct = ' ✗'

  # Plot the image
  plt.subplot(n_rows, n_cols, i+1)
  plt.imshow(np.transpose(images[0][i].numpy(), (1, 2, 0)))
  plt.title(text_true + text_pred + text_correct + text_prob)
  plt.axis('off')

plt.subplots_adjust(wspace=.01, hspace=.35)
plt.savefig("predict.png")