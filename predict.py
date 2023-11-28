from model.small_nn import NeuralNetwork
from config.config import *
from utils.data_preparation import data_prep

import torch

# Load data for make predictions
_, _, test_data = data_prep()
x, y = test_data[0][0], test_data[0][1]

# Loading the model
model = NeuralNetwork()
model.load_state_dict(torch.load("path/to/model.pt"))

model.eval()
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}" | Actual: "{actual}"')