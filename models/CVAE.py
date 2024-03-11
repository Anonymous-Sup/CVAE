import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import sys
sys.path.append('../normalizing-flows')
from normflows.flows import Planar, Radial, MaskedAffineFlow, BatchNorm
from normflows import nets
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
        
        self.flows = Flows(self.flow_input_dim)

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

        theta, logjcobin = self.flows(flow_input)
    
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

class SimpleFlowModel(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        ld = 0.0
        for flow in self.flows:
            z, ld_ = flow(z)
            ld += ld_

        return z, ld

class BinaryTransform:
    def __init__(self, thresh=0.5):
        self.thresh = thresh

    def __call__(self, x):
        return (x > self.thresh).type(x.type())


class ColourNormalize:
    def __init__(self, a=0.0, b=0.0):
        self.a = a
        self.b = b

    def __call__(self, x):
        return (self.b - self.a) * x / 255 + self.a


class Flows(nn.Module):
    def __init__(self, input_dim, flow_type='Planar', K=10):
        super().__init__()
        if flow_type== "Planar":
            flows = SimpleFlowModel([Planar((input_dim,)) for k in range(K)])
        elif flow_type == "Radial":
            flows = SimpleFlowModel([Radial((input_dim,)) for k in range(K)])
        elif flow_type == "RealNVP":
            b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(input_dim)])
            flows = []
            for i in range(K):
                s = nets.MLP([input_dim, 8, input_dim])
                t = nets.MLP([input_dim, 8, input_dim])
                if i % 2 == 0:
                    flows += [MaskedAffineFlow(b, t, s)]
                else:
                    flows += [MaskedAffineFlow(1 - b, t, s), BatchNorm()]
            flows = SimpleFlowModel(
                flows[:-1]
            )  # Remove last Batch Norm layer to allow arbitrary output
        self.flows = flows

    def forward(self, x):
        theta, logjcobin = self.flows(x)
        return theta, logjcobin
    
# add union test code for this py
if __name__ == '__main__':
    
    from thop import profile
    
    print('========Testing VAE========')
    # train(64, 512)
    vae = VAE(
        encoder_layer_sizes=[1280, 256],
        latent_size=12,
        decoder_layer_sizes=[256, 1280]
    )
    print(vae)

    # get FLOPs and Parameters
    input = torch.randn(64, 1280)
    flops, params = profile(vae, inputs=(input, torch.randn(64, 1280)))
    print('FLOPs: {:.2f}M, Params: {:.2f}M'.format(flops/1e6, params/1e6))

    
