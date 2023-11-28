import torch

# #Loading the model
# model = NeuralNetwork()
# model.load_state_dict(torch.load("model.pth"))

# #Making predictions with the model
# classes = ['normal', 'pneumonia']

# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')