import torch
import torch.nn as nn

from torch.nn import functional as F
from models.adapters import SparseBattery


class SinpleVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, zc_dim, zs_dim, n_layers=0, leak_relu_slope=0.0, bn=False):
        super().__init__()

        self.ac_fn = 'leaky_relu'
        self.bn = bn

        if self.ac_fn == 'relu':
            self.ac_fn = nn.ReLU()
        elif self.ac_fn == 'leaky_relu':
            self.ac_fn = nn.LeakyReLU(leak_relu_slope)
        elif self.ac_fn == 'elu':
            self.ac_fn = nn.ELU()
        elif self.ac_fn == 'gelu':
            self.ac_fn = nn.GELU()

        self.zc_dim = zc_dim
        self.zs_dim = zs_dim
        self.z_dim = zc_dim + zs_dim

        self.encoder =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout())
        
        self.fc_mu = nn.Sequential(nn.Linear(hidden_dim, self.z_dim))
        self.fc_logvar = nn.Sequential(nn.Linear(hidden_dim, self.z_dim))

        self.decoder = nn.Sequential(nn.Linear(self.z_dim, hidden_dim),
                                nn.BatchNorm1d(hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, input_dim))
        

        self.u_embedding = SparseBattery(num_adapters=128, c_in=input_dim, c_out=zs_dim, usebias=True)
        self.zs_embedding = nn.Sequential(nn.Linear(zs_dim * 2, zs_dim))
        
        # self.i2t_projector = nn.Linear(zc_dim, zc_dim)
        self.i2t_projector = nn.Linear(zc_dim, 512)
        # self.encoder.apply(weights_init_kaiming)
        # self.decoder.apply(weights_init_kaiming)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = track
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track
    
    # def extract_feature(self, x, track_bn=False):
    #     self.track_bn_stats(track_bn)

    #     h = self.encoder(x)

    #     mu, log_var = self.fc_mu(h), self.fc_logvar(h)
        
    #     if self.training:
    #         z = self.reparameterize(mu, log_var)
    #     else:
    #         z = mu

    #     z_idx = 0
    #     z_s = z[:, z_idx: z_idx + self.zs_dim]

    #     z_idx += self.zs_dim
    #     z_c = z[:, z_idx: z_idx + self.zc_dim]

    #     return z, z_s, z_c, mu, log_var
    
    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_logvar(h)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        
        z_idx = 0
        z_c = z[:, z_idx: z_idx + self.zc_dim]

        z_idx += self.zc_dim
        z_s = z[:, z_idx: z_idx + self.zs_dim]
    
        gate, U = self.u_embedding(x)
        
        newz_s = self.zs_embedding(torch.cat([z_s, U], dim=1))

        return h, z, z_c, z_s, newz_s, U, mu, log_var

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, track_bn=False):

        if self.training:
            self.track_bn_stats(track_bn)
        
        h, z, z_c, z_s, newz_s, U, mu, log_var = self.encode(x)

        new_z = torch.cat([z_c, newz_s], dim=1)

        recon_x = self.decode(new_z)
        
        return h, mu, log_var, z_c, z_s, U, newz_s, new_z, recon_x
    
    def i2t_projection(self, z_c):
        return self.i2t_projector(z_c)
    
    def load_param(self, param_dict, ignore_i2t=False):
        for i in self.state_dict():
            if i in param_dict.keys():
                if ignore_i2t:
                    if 'i2t_projector' in i:
                        print("Ignores parameter: ", i)
                        continue
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])

class SinpleVAE_2Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, zc_dim, zs_dim, n_layers=0, leak_relu_slope=0.0, bn=False):
        super().__init__()

        self.ac_fn = 'leaky_relu'
        self.bn = bn

        if self.ac_fn == 'relu':
            self.ac_fn = nn.ReLU()
        elif self.ac_fn == 'leaky_relu':
            self.ac_fn = nn.LeakyReLU(leak_relu_slope)
        elif self.ac_fn == 'elu':
            self.ac_fn = nn.ELU()
        elif self.ac_fn == 'gelu':
            self.ac_fn = nn.GELU()

        self.zc_dim = zc_dim
        self.zs_dim = zs_dim
        self.z_dim = zc_dim + zs_dim

        self.encoder_zc =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout())
    
        self.fc_mu_zc = nn.Sequential(nn.Linear(hidden_dim, zc_dim))
        self.fc_logvar_zc = nn.Sequential(nn.Linear(hidden_dim, zc_dim))

        self.encoder_zs = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout())
        
        self.fc_mu_zs = nn.Sequential(nn.Linear(hidden_dim, zs_dim))
        self.fc_logvar_zs = nn.Sequential(nn.Linear(hidden_dim, zs_dim))

        self.decoder = nn.Sequential(nn.Linear(self.z_dim, hidden_dim),
                                nn.BatchNorm1d(hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, input_dim))
        

        self.u_embedding = SparseBattery(num_adapters=128, c_in=input_dim, c_out=zs_dim, usebias=True)
        self.zs_embedding = nn.Sequential(nn.Linear(zs_dim * 2, zs_dim))

        self.i2t_projector = nn.Linear(zc_dim, 512)

        # if use_orthogonality:
        #     self.orthog_linear_zc = nn.Linear(zc_dim, zc_dim)
        # self.encoder.apply(weights_init_kaiming)
        # self.decoder.apply(weights_init_kaiming)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def track_bn_stats(self, track):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = track
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = track
    
    # def extract_feature(self, x, track_bn=False):
    #     self.track_bn_stats(track_bn)

    #     h = self.encoder(x)

    #     mu, log_var = self.fc_mu(h), self.fc_logvar(h)
        
    #     if self.training:
    #         z = self.reparameterize(mu, log_var)
    #     else:
    #         z = mu

    #     z_idx = 0
    #     z_s = z[:, z_idx: z_idx + self.zs_dim]

    #     z_idx += self.zs_dim
    #     z_c = z[:, z_idx: z_idx + self.zc_dim]

    #     return z, z_s, z_c, mu, log_var
    
    def encode(self, x):
        h_c = self.encoder_zc(x)
        mu_c, log_var_c = self.fc_mu_zc(h_c), self.fc_logvar_zc(h_c)

        # if self.training:
        #     z = self.reparameterize(mu, log_var)
        # else:
        #     z = mu
        z_c = self.reparameterize(mu_c, log_var_c)
        
        h_s = self.encoder_zs(x)
        mu_s, log_var_s = self.fc_mu_zs(h_s), self.fc_mu_zs(h_s)
        z_s = self.reparameterize(mu_s, log_var_s)

        gate, U = self.u_embedding(x)
        
        newz_s = self.zs_embedding(torch.cat([z_s, U], dim=1))

        return h_c, mu_c, log_var_c, z_c, h_s, mu_s, log_var_s, z_s, U, newz_s

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, track_bn=False):
        if self.training:
            self.track_bn_stats(track_bn)
        
        h_c, mu_c, log_var_c, z_c, h_s, mu_s, log_var_s, z_s, U, newz_s = self.encode(x)

        x_pre = torch.cat([h_c, h_s], dim=1)
        mean = torch.cat([mu_c, mu_s], dim=1)
        log_var = torch.cat([log_var_c, log_var_s], dim=1)
        
        new_z = torch.cat([z_c, newz_s], dim=1)

        recon_x = self.decode(new_z)

        return x_pre, mean, log_var, z_c, z_s, U, newz_s, new_z, recon_x
    
    def i2t_projection(self, z_c):
        return self.i2t_projector(z_c)
    

    def load_param(self, param_dict, ignore_i2t=True):
        for i in self.state_dict():
            if i in param_dict.keys():
                if ignore_i2t:
                    if 'i2t_projector' in i:
                        print("Ignores parameter: ", i)
                        continue
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])