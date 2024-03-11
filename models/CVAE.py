import torch
import torch.nn as nn
from torch.distributions.normal import Normal

# import sys
# # from utils import idx2onehot

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, feat_fusion=True):

        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.feature_fusion = feat_fusion

        self.domain_embeding = nn.Sequential()
        self.domain_embeding.add_module('domain_embeding_layer1', nn.Linear(encoder_layer_sizes[0], latent_size))
        self.domain_embeding.add_module('domain_embeding_activate', nn.ReLU())

        if self.feature_fusion:
            self.fusion = nn.Sequential()
            self.fusion.add_module('fusion_layer1', nn.Linear(latent_size*2, latent_size))
            self.fusion.add_module('fusion_activate', nn.ReLU())
            self.flow_input_dim = latent_size
        else:
            self.flow_input_dim = latent_size*2

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size)
        
        self.flow = Flow(self.flow_input_dim)

    def forward(self, x, domain_index_feat):

        if x.dim() > 2:
            x = x.view(-1, 1280)
        
        domain_index_feat= self.domain_embeding(domain_index_feat)
        
        means, log_var = self.encoder(x)

        """
        Todo: Here need a flows model as a prior of z
        """
        z_0 = self.reparameterize(means, log_var)
        
        # (b, size) -> (b, 2*size)
        flow_input = torch.cat((z_0, domain_index_feat), dim=-1)

        if self.feature_fusion:
            # (b, 2*size) -> (b, size)
            flow_input = self.fusion(flow_input)

        theta, logjcobin = self.flow(flow_input)
    
        # KL Loss=−(−1/2(log(2pai)+σ^2)−Jacobian)
        # u tends to 0 and σ tends to 1
        # kl_loss = 0.5 * torch.sum(theta**2, dim=-1) - torch.sum(logjcobin, dim=-1)

        recon_x = self.decoder(z_0)

        return recon_x, means, log_var, z_0, theta, logjcobin

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        # self.conditional = conditional
        # if self.conditional:
        #     layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):

        # if self.conditional:
        #     c = idx2onehot(c, n=10)
        #     x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):

        super().__init__()

        self.MLP = nn.Sequential()

        # self.conditional = conditional
        # if self.conditional:
        #     input_size = latent_size + num_labels
        # else:
            # input_size = latent_size
        
        input_size = latent_size

        # (12+256, [256,512])
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):

        # if self.conditional:
        #     c = idx2onehot(c, n=10)
        #     z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)

        return x



class Flow(nn.Module):
    def __init__(self, input_dim):

        super().__init__()

        self.f1 = nn.Squential()
        self.f1.add_module('f1_layer1', nn.Linear(input_dim, input_dim))
        self.f1.add_module('f1_activate', nn.LeakyReLU())
        self.f2 = nn.Squential()
        self.f2.add_module('f1_layer2', nn.Linear(input_dim, input_dim))
        self.f2.add_module('f1_activate', nn.LeakyReLU())

    def forward(self, x):
        theta = self.f1(x)
        logjcobin = self.f2(x)
        return theta, logjcobin
    
# add union test code for this py
if __name__ == '__main__':
    
    from thop import profile
    
    print('========Testing VAE========')
    # train(64, 512)
    vae = VAE(
        encoder_layer_sizes=[1280, 256],
        latent_size=12,
        decoder_layer_sizes=[256, 1280],
        conditional=False,
        num_labels=0)
    print(vae)

    # get FLOPs and Parameters
    input = torch.randn(64, 1280)
    flops, params = profile(vae, inputs=(input,))
    print('FLOPs: {:.2f}M, Params: {:.2f}M'.format(flops/1e6, params/1e6))

    
