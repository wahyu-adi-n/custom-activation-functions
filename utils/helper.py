import torch.nn as nn

# Replace ReLU with Other AFs in the model
def replace_afs(module, func):
    for child_name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, child_name, func)
        else:
            replace_afs(child, func)