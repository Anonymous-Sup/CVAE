import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter


__all__ = ['Classifier', 'NormalizedClassifier']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        m.bias.requires_grad_(False)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        init.normal_(self.classifier.weight.data, std=0.001)
        # init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        y = self.classifier(x)

        return y
        

class NormalizedClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_classes, feature_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5) 

    def forward(self, x):
        w = self.weight  

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        return F.linear(x, w)
    

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
# Low dim feature to hideen to classifer
class MLPClassBlock(nn.Module):
    def __init__(self, feature_dim, num_classes, droprate=0.2):
        super(MLPClassBlock, self).__init__()

        leaky_relu_slope = 0.2         
        hidden_dim_list = [64, 128, 256, 512]
        print('Using MLPClassBlock with hiidem_dim_list:', hidden_dim_list)
        
        layers = []
        pre_dim = feature_dim
        for dim in hidden_dim_list:
            layers.append(nn.Linear(pre_dim, dim, bias=False))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.LeakyReLU(leaky_relu_slope))
            if droprate>0:
                layers.append(nn.Dropout(p=droprate))
            pre_dim = dim
            
        self.pre_embedding = nn.Sequential(*layers)
        self.pre_embedding.apply(weights_init_kaiming)

        classifier = []
        classifier.append(nn.Linear(hidden_dim_list[-1], num_classes, bias=False))
        self.classifier = nn.Sequential(*classifier)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.pre_embedding(x)
        x = self.classifier(x)
        return x