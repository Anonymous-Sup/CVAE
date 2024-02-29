import torch
import torch.nn as nn




class KLD_loss(nn.Module):
    def __init__(self):
        super(KLD_loss, self).__init__()

    def forward(self, mean, log_var):
        loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return loss.mean()


class BCE_loss(nn.Module):
    def __init__(self, feature_dim):
        super(BCE_loss, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, recon_x, x):
        loss = nn.functional.binary_cross_entropy(
            recon_x.view(-1, self.feature_dim), x.view(-1, self.feature_dim), reduction='sum')
        return loss.mean()


