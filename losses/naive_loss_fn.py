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
        self.bce_with_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
    def forward(self, recon_x ,x):
        # torch.nn.functional.binary_cross_entropy 
        # and torch.nn.BCELoss are unsafe to autocast.
        loss = self.bce_with_logits_loss(recon_x.view(-1, self.feature_dim), x.view(-1, self.feature_dim))
        return loss.mean()

class MSE_loss(nn.Module):
    def __init__(self, feature_dim):
        super(MSE_loss, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, recon_x, x):
        # normalize recon_x
        # recon_x /= recon_x.norm(dim=-1, keepdim=True)
        # x /= x.norm(dim=-1, keepdim=True)
        # torch.nn.functional.sigmoid(recon_x)
        loss = nn.functional.mse_loss(recon_x.view(-1, self.feature_dim), x.view(-1, self.feature_dim), reduction='sum')
        return loss.mean()

class MMD_loss(nn.Module):
    def __init__(self, sigma=1.0):
        super(MMD_loss, self).__init__()
        self.sigma = sigma

    def forward(self, z):
        batch_size = z.size(0)
        kernel_matrix = self.gaussian_kernel(z, z, self.sigma)
        loss = (kernel_matrix.sum() - kernel_matrix.diag().sum()) / (batch_size * (batch_size - 1))
        return loss
    
    def gaussian_kernel(self, x1, x2, sigma=1.0):
        dist_matrix = torch.cdist(x1, x2, p=2)
        kernel_matrix = torch.exp(-dist_matrix / (2 * sigma ** 2))
        return kernel_matrix


class MAE_loss(nn.Module):
    def __init__(self, feature_dim):
        super(MAE_loss, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, recon_x, x):
        loss = nn.functional.l1_loss(recon_x.view(-1, self.feature_dim), x.view(-1, self.feature_dim), reduction='sum')
        return loss.mean()
    
class SmoothL1_loss(nn.Module):
    def __init__(self, feature_dim, beta=1.0):
        super(SmoothL1_loss, self).__init__()
        self.feature_dim = feature_dim
        self.beta = beta

    def forward(self, recon_x, x):
        loss = nn.functional.smooth_l1_loss(recon_x.view(-1, self.feature_dim), x.view(-1, self.feature_dim), beta=self.beta, reduction='sum')
        return loss.mean()
    
class Pearson_loss(nn.Module):
    def __init__(self, feature_dim):
        super(Pearson_loss, self).__init__()
        self.feature_dim = feature_dim

    def forward(self, recon_x, x):
        recon_x = recon_x.view(-1, self.feature_dim)
        x = x.view(-1, self.feature_dim)
        
        vx = x - x.mean(dim=1, keepdim=True)
        vy = recon_x - recon_x.mean(dim=1, keepdim=True)

        corr = (vx * vy).sum(dim=1) / (vx.norm(dim=1) * vy.norm(dim=1))
        loss = 1 - corr
        return loss.mean()
    
