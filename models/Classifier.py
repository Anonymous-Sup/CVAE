import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
from torch.nn.utils.weight_norm import WeightNorm

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
    def __init__(self, feature_dim, num_classes, usebias=False):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes, bias=usebias)
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


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 
        # scores.shape = (batch_size, num_classes)?
        return scores

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
# Low dim feature to hideen to classifer
class MLPClassBlock(nn.Module):
    def __init__(self, feature_dim, num_classes, droprate=0):
        super(MLPClassBlock, self).__init__()

        leaky_relu_slope = 0.2         
        # hidden_dim_list = [64, 128, 256, 512]
        hidden_dim_list = [256, 512, 1024]
        print('Using MLPClassBlock with dimenssion flow: {}-->{}-->{}'.format(feature_dim, hidden_dim_list, num_classes))
        
        layers = []
        pre_dim = feature_dim
        for dim in hidden_dim_list:
            layers.append(nn.Linear(pre_dim, dim, bias=False))
            # layers.append(nn.BatchNorm1d(dim))
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