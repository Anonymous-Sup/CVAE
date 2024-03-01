import torch
import torch.nn as nn

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

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, feat_fusion=False):

        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.feature_fusion = feat_fusion

        self.domain_embeding = nn.squential()
        self.domain_embeding.add_module('domain_embeding_layer1', nn.Linear(encoder_layer_sizes[0], latent_size))
        self.domain_embeding.add_module('domain_embeding_activate', nn.ReLU())

        if self.feature_fusion:
            self.fusion = nn.sequential()
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
            x = x.view(-1, 512)
        
        domain_index_feat= self.domain_embeding(domain_index_feat)
        
        means, log_var = self.encoder(x)

        """
        Todo: Here need a flows model as a prior of z
        """
        z = self.reparameterize(means, log_var)
        
        # (b, size) -> (b, 2*size)
        flow_input = torch.cat((z, domain_index_feat), dim=-1)

        if self.feature_fusion:
            # (b, 2*size) -> (b, size)
            flow_input = self.fusion(flow_input)

        theta, logjcobin = flow(flow_input)

        recon_x = self.decoder(z)

        return recon_x, means, log_var, z, theta, logjcobin

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

# add union test code for this py
if __name__ == '__main__':
    
    from thop import profile
    
    print('========Testing VAE========')
    # train(64, 512)
    vae = VAE(
        encoder_layer_sizes=[512, 256],
        latent_size=12,
        decoder_layer_sizes=[256, 512],
        conditional=False,
        num_labels=0)
    print(vae)

    # get FLOPs and Parameters
    input = torch.randn(64, 512)
    flops, params = profile(vae, inputs=(input,))
    print('FLOPs: {:.2f}M, Params: {:.2f}M'.format(flops/1e6, params/1e6))

    
