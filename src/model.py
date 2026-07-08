import torch
import torch.nn as nn
from torchvision import models

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def build_model(device):
    # Set the model to be loaded as pretrained
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
    
    # Freeze the weights to only train the classifier layer
    for param in resnet.parameters(): 
        param.requires_grad = False     

    # Modify the final layer to output the 102 categories
    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, 102),
        nn.Softmax(dim=1) 
    )

    resnet.to(device)
    return resnet
