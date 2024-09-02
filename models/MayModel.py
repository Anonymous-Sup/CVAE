import torch
import torch.nn as nn

from torch.nn import functional as F
from models.adapters import SparseBattery
from utils import weights_init_kaiming

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, number_layers=4, leak_relu_slope=0.2, bn=False):
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

class TransformerReIDProjection(nn.Module):
    def __init__(self, input_dim, projection_dim=1280, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerReIDProjection, self).__init__()
        
        # Linear layer to project z_c to the projection dimension
        self.fc_input = nn.Linear(input_dim, projection_dim)

        # Positional encoding
        # self.positional_encoding = nn.Parameter(torch.randn(1, projection_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=projection_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection layer
        self.fc_proj = nn.Linear(projection_dim, projection_dim)
        self.bn_proj = nn.BatchNorm1d(projection_dim)
    
    def forward(self, x):
        # Project input to projection dimension
        x = self.fc_input(x)

        # Add positional encoding to input
        # x = x + self.positional_encoding
        
         # Transformer encoding
        x = x.unsqueeze(1).permute(1, 0, 2)  # Transformer expects input shape (seq_len, batch, dim)
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Revert back to shape (batch, dim)
        
        # Final projection
        x = self.fc_proj(x)
        x_bn = self.bn_proj(x)
        x_bn = F.relu(x_bn)
        
        if self.training:
            return x_bn
        else:
            return x

class SinpleVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, zc_dim, zs_dim, style_num=0, n_layers=0, leak_relu_slope=0.2, bn=False, projection_type=None):
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

        self.style_num = style_num

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
        
        if self.style_num > 0:
            self.zs_embedding = nn.Sequential(nn.Linear(zs_dim * 3, zs_dim))
        else:
            self.zs_embedding = nn.Sequential(nn.Linear(zs_dim * 2, zs_dim))
        
        self.projection_type = projection_type #  'Linear1280+CLS' 
        
        if self.projection_type == 'Linear+CLS':
            self.reid_output_dim = self.zc_dim
            self.reid_projector = nn.Linear(self.zc_dim, self.reid_output_dim, bias=False)
        elif self.projection_type == 'MLP+CLS':
            self.reid_output_dim = self.zc_dim
            self.reid_projector = MLP(self.zc_dim, 256, self.reid_output_dim)  # no bias term
        
        elif self.projection_type == 'Linear1280+CLS':
            self.reid_output_dim = 1280
            self.reid_projector = nn.Linear(self.zc_dim, self.reid_output_dim, bias=False)
        elif self.projection_type == 'MLP1280+CLS':
            self.reid_output_dim = 1280
            self.reid_projector = MLP(self.zc_dim, 256, self.reid_output_dim) # no bias term
        
        elif self.projection_type == 'MLP768+CLS':
            self.reid_output_dim = 768
            self.reid_projector = MLP(self.zc_dim, 256, self.reid_output_dim) # no bias term
        elif self.projection_type == 'Transforer1280+CLS':
            self.reid_output_dim = 1280
            self.reid_projector = TransformerReIDProjection(self.zc_dim, self.reid_output_dim)
        elif self.projection_type == None:
            print("============Warning! No projection layer is used in Training!=================")
            print("============Warning! Using Defalt Linear 1280 to intialize!=================")
            self.reid_output_dim = 1280
            self.reid_projector = nn.Linear(self.zc_dim, self.reid_output_dim, bias=False)
        else:
            raise ValueError("Invalid projection type {}", self.projection_type)

        # i2t_input_dim = zc_dim
        i2t_input_dim = self.reid_output_dim
        self.cls_input_dim = 1280
        self.i2t_projector = nn.Linear(i2t_input_dim, self.cls_input_dim)

        if self.style_num > 0:
            self.style_embedding = nn.Embedding(self.style_num, zs_dim)
        
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
    
    def encode(self, x, style=None):
        
        if self.style_num > 0 and style is not None:
            style_emb = self.style_embedding(style)

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

        if self.style_num > 0 and style is not None:
            U = torch.cat([U, style_emb], dim=1)
            
        newz_s = self.zs_embedding(torch.cat([z_s, U], dim=1))

        return h, z, z_c, z_s, newz_s, U, mu, log_var

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, style=None, track_bn=False):

        if self.training:
            self.track_bn_stats(track_bn)
        
        h, z, z_c, z_s, newz_s, U, mu, log_var = self.encode(x, style)

        new_z = torch.cat([z_c, newz_s], dim=1)

        recon_x = self.decode(new_z)
        
        return h, mu, log_var, z_c, z_s, U, newz_s, new_z, recon_x
    
    # def i2t_projection(self, z_c):
    #     return self.i2t_projector(z_c)
    
    # def reid_projection(self, z_c):
    #     return self.reid_projector(z_c)
    
    def load_param(self, param_dict, ignore_i2t=False, ignore_reid=False, ignore_encoder=False):
        for i in self.state_dict():
            if i in param_dict.keys():
                print("Loading all parameter: ", i)
                if ignore_i2t:
                    if 'i2t_projector' in i:
                        print("Ignores parameter: ", i)
                        continue
                if ignore_reid:
                    if 'reid_projector' in i:
                        print("Ignores parameter: ", i)
                        continue
                elif 'reid_projector' in i:
                    print("Loading parameter: ", i)
                if ignore_encoder:
                    if 'fc' in i or 'encoder' in i:
                        print("Ignores parameter: ", i)
                        continue
                    
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                print("Missing and Initializing parameter: ", i)

class SinpleVAE_2Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, zc_dim, zs_dim, style_num=0, n_layers=0, leak_relu_slope=0.0, bn=False, projection_type=None):
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

        self.style_num = style_num

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
        
        if self.style_num > 0:
            self.zs_embedding = nn.Sequential(nn.Linear(zs_dim * 3, zs_dim))
        else:
            self.zs_embedding = nn.Sequential(nn.Linear(zs_dim * 2, zs_dim))

        self.projection_type =  projection_type #  'Linear1280+CLS'

        if self.projection_type == 'Linear+CLS':
            self.reid_output_dim = self.zc_dim
            self.reid_projector = nn.Linear(self.zc_dim, self.reid_output_dim, bias=False)
        elif self.projection_type == 'MLP+CLS':
            self.reid_output_dim = self.zc_dim
            self.reid_projector = MLP(self.zc_dim, 256, self.reid_output_dim)  # no bias term

        elif self.projection_type == 'Linear1280+CLS':
            self.reid_output_dim = 1280
            self.reid_projector = nn.Linear(self.zc_dim, self.reid_output_dim, bias=False)
        elif self.projection_type == 'MLP1280+CLS':
            self.reid_output_dim = 1280
            self.reid_projector = MLP(self.zc_dim, 256, self.reid_output_dim) # no bias term

        elif self.projection_type == 'MLP768+CLS':
            self.reid_output_dim = 768
            self.reid_projector = MLP(self.zc_dim, 256, self.reid_output_dim) # no bias term
        elif self.projection_type == 'Transforer1280+CLS':
            self.reid_output_dim = 1280
            self.reid_projector = TransformerReIDProjection(self.zc_dim, self.reid_output_dim)
        elif self.projection_type == None:
            print("============Warning! No projection layer is used in Training!=================")
            print("============Warning! Using Defalt Linear 1280 to intialize!=================")
            self.reid_output_dim = 1280
            self.reid_projector = nn.Linear(self.zc_dim, self.reid_output_dim, bias=False)
        else:
            raise ValueError("Invalid projection type {}", self.projection_type)

        # i2t_input_dim = zc_dim
        i2t_input_dim = self.reid_output_dim
        self.cls_input_dim = 1280
        self.i2t_projector = nn.Linear(i2t_input_dim, self.cls_input_dim)

        if self.style_num > 0:
            self.style_embedding = nn.Embedding(self.style_num, zs_dim)

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
    
    def encode(self, x, style=None):
        
        if self.style_num>0 and style is not None:
            style_emb = self.style_embedding(style)

        h_c = self.encoder_zc(x)
        mu_c, log_var_c = self.fc_mu_zc(h_c), self.fc_logvar_zc(h_c)
        
        h_s = self.encoder_zs(x)
        mu_s, log_var_s = self.fc_mu_zs(h_s), self.fc_mu_zs(h_s)

        if self.training:
            z_c = self.reparameterize(mu_c, log_var_c)
            z_s = self.reparameterize(mu_s, log_var_s)
        else:
            z_c = mu_c
            z_s = mu_s

        gate, U = self.u_embedding(x)
        
        if self.style_num > 0 and style is not None:
            U = torch.cat([U, style_emb], dim=1)

        newz_s = self.zs_embedding(torch.cat([z_s, U], dim=1))

        return h_c, mu_c, log_var_c, z_c, h_s, mu_s, log_var_s, z_s, U, newz_s

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, style=None, track_bn=False):
        if self.training:
            self.track_bn_stats(track_bn)
        
        h_c, mu_c, log_var_c, z_c, h_s, mu_s, log_var_s, z_s, U, newz_s = self.encode(x, style)

        x_pre = torch.cat([h_c, h_s], dim=1)
        mean = torch.cat([mu_c, mu_s], dim=1)
        log_var = torch.cat([log_var_c, log_var_s], dim=1)
        
        new_z = torch.cat([z_c, newz_s], dim=1)

        recon_x = self.decode(new_z)

        return x_pre, mean, log_var, z_c, z_s, U, newz_s, new_z, recon_x
    
    # def i2t_projection(self, z_c):
    #     return self.i2t_projector(z_c)
    
    # def reid_projection(self, z_c):
    #     return self.reid_projector(z_c)

    def load_param(self, param_dict, ignore_i2t=False, ignore_reid=False, ignore_encoder=False):
        for i in self.state_dict():
            if i in param_dict.keys():
                print("Loading all parameter: ", i)
                if ignore_i2t:
                    if 'i2t_projector' in i:
                        print("Ignores parameter: ", i)
                        continue
                if ignore_reid:
                    if 'reid_projector' in i:
                        print("Ignores parameter: ", i)
                        continue
                elif 'reid_projector' in i:
                    print("Loading parameter: ", i)
                if ignore_encoder:
                    if 'fc' in i or 'encoder' in i:
                        print("Ignores parameter: ", i)
                        continue
                
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                print("Missing and Initializing parameter: ", i)