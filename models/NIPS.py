import torch
import torch.nn as nn


class NIPS(nn.Module):
    def __init__(self, VAE, FLOWs, feature_dim, hidden_dim, latent_size):
        super(NIPS, self).__init__()

        self.VAE = VAE
        self.FLOWs = FLOWs

        if hidden_dim is None:
            self.hidden_dim = [256]
            print("hidden_dim is not specified, set to be {}".format(self.hidden_dim))
        else:
            self.hidden_dim = hidden_dim

        self.domain_embeding = nn.Sequential()
        self.domain_embeding.add_module('domain_embeding_layer1', nn.Linear(feature_dim, self.hidden_dim[0]))
        self.domain_embeding.add_module('domain_embeding_activate', nn.ELU())
        self.domain_embeding.add_module('domain_embeding_layer2', nn.Linear(self.hidden_dim[0], latent_size))
        
        
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer1', nn.Linear(latent_size*2, latent_size))
        self.fusion.add_module('fusion_activate', nn.ELU())
        self.fusion.add_module('fusion_layer2', nn.Linear(latent_size, latent_size))


        # self.domain_embeding.apply(self.init_weights)
        # self.fusion.apply(self.init_weights)

    def forward(self, x, domain_index):
        
        domain_index_feat= self.domain_embeding(domain_index)
        domain_index_feat = self.normalization(domain_index_feat)

        means, log_var = self.VAE.encoder(x)
        z_0 = self.VAE.reparameterize(means, log_var)

        flow_input = torch.cat((z_0, domain_index_feat), dim=-1)
        z_1 = self.fusion(flow_input)

        theta, logjcobin = self.FLOWs(z_1)

        recon_x = self.VAE.decoder(z_0)

        return recon_x, means, log_var, z_0, z_1, theta, logjcobin
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def normalization(self, x):
        # x_normalized = x / x.norm(dim=1, keepdim=True)
        # 手动计算均值和标准差进行归一化
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        x_normalized = (x - mean) / std
        return x_normalized