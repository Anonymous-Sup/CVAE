import torch
import torch.nn as nn
from utils import weights_init_kaiming, idx2onehot
from models.adapters import SparseBattery

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, number_layers=3, leak_relu_slope=0.2, bn=False):
        super(MLP, self).__init__()

        layers = []
        for l in range(number_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
                layers.append(nn.LeakyReLU(leak_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                layers.append(nn.LeakyReLU(leak_relu_slope))
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        layers.append(nn.LeakyReLU(leak_relu_slope))
        # if bn:
        #     layers.append(nn.BatchNorm1d(output_dim))

        self.MLP = nn.Sequential(*layers)

        self.MLP.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.MLP(x)
        return x

# input dim is two feature of 64,64 , the output is 64,64
class BilinearPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BilinearPooling, self).__init__()
        self.bilinear = nn.Bilinear(input_dim, input_dim, output_dim, bias=False)
        self.bilinear.apply(weights_init_kaiming)

    def forward(self, x1, x2):
        x = self.bilinear(x1, x2)
        return x


class SparseLatentAdapter(nn.Module):
    def __init__(self, num_adapters, c_in, c_out):
        super(SparseLatentAdapter, self).__init__()
        
        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.parallel_adapter = SparseBattery(num_adapters, c_in, c_out)

    def forward(self, x):
        # this is f_0(x), which is a shared linear
        y = self.linear(x)
    
        # this is the sum of K adapters with gate and indendepnt 1*1 linear
        # gate.shape = (batchsize, num_adapters)
        # out.shape = (batchsize, 64)
        gate, out = self.parallel_adapter(x)

        # this f_0(x) + sum_k(g_k(x) * f_k(x))
        y = y + out
        y = self.bn(y)

        return gate, y

class SparseLatentAdapterBlock(nn.Module):
    def __init__(self, num_adapters, in_features, out_features):
        super(SparseLatentAdapterBlock, self).__init__()
        self.sla1 = SparseLatentAdapter(num_adapters, in_features, out_features)
        self.sla2 = SparseLatentAdapter(num_adapters, out_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, y = self.sla1(x)
        y = self.relu(y)
        gate, y = self.sla2(y)
        y = self.relu(y)
        return gate, y
    


class MLP_Adapter(nn.Module):
    def __init__(self, num_adapters, input_dim, hidden_dim, output_dim, number_layers=3, bn=False):
        super(MLP_Adapter, self).__init__()

        self.bn = bn
        self.layers = nn.ModuleList()
        for l in range(number_layers):
            if l == 0:
                self.layers.append(SparseLatentAdapterBlock(num_adapters, input_dim, hidden_dim))
            else:
                self.layers.append(SparseLatentAdapterBlock(num_adapters, hidden_dim, hidden_dim))

        self.layers.append(SparseLatentAdapterBlock(num_adapters, hidden_dim, output_dim))

        if self.bn:
            self.batchnrom = nn.BatchNorm1d(output_dim)

        self.apply(weights_init_kaiming)

    def forward(self, x):
        for layer in self.layers[:-1]:
            _, x = layer(x)

        gate, x = self.layers[-1](x)

        if self.bn:
            x = self.batchnrom(x)

        return gate, x

class NIPS(nn.Module):
    def __init__(self, VAE, FLOWs, feature_dim, hidden_dim, out_dim, latent_size, only_x=False, use_centroid=False, latent_z='z_0'):
        super(NIPS, self).__init__()

        self.VAE = VAE
        self.FLOWs = FLOWs

        self.latent_size = latent_size
        self.only_x_input = only_x
        self.use_centroid = use_centroid

        if hidden_dim is None:
            self.hidden_dim = 768 # default hidden_dim 768
            print("hidden_dim is not specified, set to be {}".format(self.hidden_dim))
        else:
            self.hidden_dim = hidden_dim
        
        self.out_dim = out_dim 

        # self.domain_adapters = SparseLatentAdapter(num_adapters=128, c_in=feature_dim, c_out=out_dim)
        self.domain_adapters = MLP_Adapter(num_adapters=128, input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=out_dim, number_layers=4, bn=True)

        self.norm = self.normalize_l2

        self.latent_z = latent_z

    def forward(self, x, norm=True):
        
        gate, domain_feature = self.domain_adapters(x)
        # count the value of gate at the second dim that not zero, and output the value for each image in the batch(dim=0)
        # non_zero_counts = (gate != 0).sum(dim=1)
        # print("non_zero_counts: ", non_zero_counts)
        
        if torch.isnan(domain_feature).any():
            print("domain_feature has nan")
        
        if norm:
            domain_feature_norm = self.norm(domain_feature)
        else:
            domain_feature_norm = domain_feature

        x_pre, means, log_var = self.VAE.encoder(x) # (64, latentsize)

        if norm:
            x_proj_norm = self.norm(x_pre)
        else:
            x_proj_norm = x_pre

        z_0 = self.VAE.reparameterize(means, log_var)   # (64, latentsize)

        if self.latent_z == 'z_0':
            recon_x = self.VAE.decoder(z_0)
        elif self.latent_z == 'x_pre':
            recon_x = self.VAE.decoder(x_proj_norm)
        else:
            raise ValueError("latent_z should be one of ['z_0', 'fuse_z', 'x_pre']")

        '''
        cat u_i with each dim of z
        '''
        # # 64, 12, 64
        # domian_dim =  int(self.hidden_dim/self.latent_size)
        # U = domain_feature_norm.view(-1, self.latent_size, domian_dim)
        # # (64, 12, 1) + (64, 12, 64) -> (64, 12, 65)
        # flow_input = torch.cat((U, x_proj_norm.unsqueeze(-1)), dim=-1)


        '''
        cat u with each dim of z
        '''
        U = domain_feature_norm.unsqueeze(1)
        U = U.repeat(1, x_proj_norm.size(1), 1)
        # (64,12) --> (64, 12, 1) + (64, 12, 64) --> (64, 12, 65)
        if self.latent_z == 'z_0':
            flow_input = torch.cat((U, z_0.unsqueeze(-1)), dim=-1)
        elif self.latent_z == 'x_pre':
            flow_input = torch.cat((U, x_proj_norm.unsqueeze(-1)), dim=-1)
        else:
            raise ValueError("latent_z should be one of ['z_0', 'fuse_z', 'x_pre']")

        if self.only_x_input:
            if self.latent_z == 'z_0':
                z_1 = z_0
            elif self.latent_z == 'x_pre':
                z_1 = x_proj_norm
        else:
            z_1 = flow_input

        theta, logjcobin = self.FLOWs(z_1)

        return recon_x, means, log_var, z_0, x_pre, x_proj_norm, z_1, theta, logjcobin, domain_feature, flow_input


    def normalization(self, x):
        # x_normalized = x / x.norm(dim=1, keepdim=True)
        # 手动计算均值和标准差进行归一化
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        x_normalized = (x - mean) / std
        return x_normalized
    
    def normalize_l2(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
        x: pytorch Variable
        Returns:
        x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x
    
    def min_max_normalization(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        if max_val - min_val == 0:
            print("Max and Min are equal in Nolmalization!")
            return tensor
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor
    
    def normalize_l2_01(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
        x: pytorch Variable
        Returns:
        x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        
        # using sigmod to normalize to [0, 1]
        # x = torch.sigmoid(x)

        # using min_max to normalize to [0, 1]
        x = self.min_max_normalization(x)
        return x
    
