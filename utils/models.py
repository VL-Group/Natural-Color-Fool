import torch
import torchvision.models as models
import torch.nn as nn
import timm

# Normalize
class Normalize(nn.Module):
    """
    mode:
        'tensorflow':convert data from [0,1] to [-1,1]
        'torch':(input - mean) / std
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], mode='tensorflow'):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 -1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x

def get_torch_model(model_name, device='cuda:0'):
    "Load Pytorch model"
    device  = torch.device(device)
    net = getattr(models, model_name)(pretrained=True)
    model = nn.Sequential(
        Normalize(mode='torch'), 
        net.eval().to(device),)
    return model


def get_timm_model(model_name, device='cuda:0'):
    device  = torch.device(device)
    model = nn.Sequential(
        Normalize(mode='torch',mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        timm.create_model(model_name, pretrained=True).eval().to(device))
    return model

def get_models(models_name_list, mode, device='cuda:0'):
    """load models with dict"""
    models = {}
    if mode == 'torch':
        for name in models_name_list:
            models[name] = get_torch_model(name, device)
    elif mode == 'timm':
        for name in models_name_list:
            models[name] = get_timm_model(name, device)
    return models
