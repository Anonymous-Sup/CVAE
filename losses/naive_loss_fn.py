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
        # torch.nn.functional.binary_cross_entropy 
        # and torch.nn.BCELoss are unsafe to autocast.
        loss = nn.functional.binary_cross_entropy_with_logits(
            recon_x.view(-1, self.feature_dim), x.view(-1, self.feature_dim), reduction='sum')
        return loss.mean()

class MSE_loss(nn.Module):
    def __init__(self, feature_dim):
        super(MSE_loss, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, recon_x, x):
        # normalize recon_x
        # recon_x /= recon_x.norm(dim=-1, keepdim=True)
        # x /= x.norm(dim=-1, keepdim=True)

        loss = nn.functional.mse_loss(recon_x.view(-1, self.feature_dim), x.view(-1, self.feature_dim), reduction='sum')
        return loss.mean()

