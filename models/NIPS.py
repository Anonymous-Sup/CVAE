import torch
import torch.nn as nn


class NIPS(nn.Module):
    def __init__(self, VAE, FLOWs, feature_dim, latent_size):
        super(NIPS, self).__init__()

        self.VAE = VAE
        self.FLOWs = FLOWs

        self.domain_embeding = nn.Sequential()
        self.domain_embeding.add_module('domain_embeding_layer1', nn.Linear(feature_dim, latent_size))
        self.domain_embeding.add_module('domain_embeding_activate', nn.ReLU())

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer1', nn.Linear(latent_size*2, latent_size))
        self.fusion.add_module('fusion_activate', nn.ReLU())


    def forward(self, x, domain_index_feat):

        domain_index_feat= self.domain_embeding(domain_index_feat)

        means, log_var = self.VAE.encoder(x)
        z_0 = self.VAE.reparameterize(means, log_var)

        flow_input = torch.cat((z_0, domain_index_feat), dim=-1)
        z_1 = self.fusion(flow_input)

        theta, logjcobin = self.FLOWs(z_1)

        recon_x = self.VAE.decoder(z_0)

        return recon_x, means, log_var, z_0, z_1, theta, logjcobin