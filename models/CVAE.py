import torch
import torch.nn as nn


def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

class VAE(nn.Module):

    def __init__(self, feature_dim, hidden_dim, output_dim, n_layers):

        super().__init__()

        self.ac_fn = 'leaky_relu'

        if self.ac_fn == 'relu':
            self.ac_fn = nn.ReLU()
        elif self.ac_fn == 'leaky_relu':
            self.ac_fn = nn.LeakyReLU(0.2)
        elif self.ac_fn == 'elu':
            self.ac_fn = nn.ELU()
    
        self.encoder = Encoder(
            feature_dim, hidden_dim, output_dim, n_layers, self.ac_fn)
        self.decoder = Decoder(
            output_dim, hidden_dim, feature_dim, n_layers, self.ac_fn)
        
        # self.encoder.apply(self.init_weights)
        # self.decoder.apply(self.init_weights)

    # remember to be fp16
    def forward(self, x, y):
        return True

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) + 1e-8
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):

    def __init__(self, input_dim, hiden_dim, out_dim, n_layers, ac_fn):

        super().__init__()

        # self.conditional = conditional
        # if self.conditional:
        #     layer_sizes[0] += num_labels
        self.MLP = nn.Sequential()  
        # at least 3 layers,         
        for i in range(n_layers):
            if i == 0:
                self.MLP.add_module(
                    name="L{:d}".format(i), module=nn.Linear(input_dim, hiden_dim))
                self.MLP.add_module(name="A{:d}".format(i), module=ac_fn)
            else:
                self.MLP.add_module(
                    name="L{:d}".format(i), module=nn.Linear(hiden_dim, hiden_dim))
                self.MLP.add_module(name="A{:d}".format(i), module=ac_fn)
        self.MLP.add_module(
            name="L{:d}".format(n_layers), module=nn.Linear(hiden_dim, out_dim))
        
        # 1280 -> 256 -> 36
        self.linear_means = nn.Linear(out_dim, out_dim)
        self.linear_log_var = nn.Linear(out_dim, out_dim)
        
        # add leak relu
        self.ac_fn = nn.LeakyReLU(0.2)
        # self.ac_fn = None

    def forward(self, x):

        x = self.MLP(x)
        
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        if self.ac_fn is not None:
            means = self.ac_fn(means)
            log_vars = self.ac_fn(log_vars)

        return x, means, log_vars


class Decoder(nn.Module):

    def __init__(self, input_dim, hiden_dim, out_dim, n_layers, ac_fn):

        super().__init__()

        self.MLP = nn.Sequential()
        
        # 12 -> 256 -> 1280
        for i in range(n_layers):
            if i == 0:
                self.MLP.add_module(
                    name="L{:d}".format(i), module=nn.Linear(input_dim, hiden_dim))
                self.MLP.add_module(name="A{:d}".format(i), module=ac_fn)
            else:
                self.MLP.add_module(
                    name="L{:d}".format(i), module=nn.Linear(hiden_dim, hiden_dim))
                self.MLP.add_module(name="A{:d}".format(i), module=ac_fn)
        
        self.MLP.add_module(
            name="L{:d}".format(n_layers), module=nn.Linear(hiden_dim, out_dim))
        

    def forward(self, z):

        # if self.conditional:
        #     c = idx2onehot(c, n=10)
        #     z = torch.cat((z, c), dim=-1)
        # self.MLP = self.MLP.to(dtype=z.dtype)
        x = self.MLP(z)
        x = x / x.norm(dim=-1, keepdim=True)

        return x



class VAE_backup(nn.Module):

    def __init__(self, flows, encoder_layer_sizes, latent_size, decoder_layer_sizes, feat_fusion=True):

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
        
        self.flows = flows 
        

    # remember to be fp16
    def forward(self, x, domain_index_feat):

        # if x.dim() > 2:
        #     x = x.view(-1, 1280)
        
        domain_index_feat= self.domain_embeding(domain_index_feat)
        means, log_var = self.encoder(x)

        z_0 = self.reparameterize(means, log_var)
        
        # (b, size) -> (b, 2*size)
        flow_input = torch.cat((z_0, domain_index_feat), dim=-1)
        if self.feature_fusion:
            # (b, 2*size) -> (b, size)
            z_1 = self.fusion(flow_input)
        
        theta, logjcobin = self.flows(z_1)
        recon_x = self.decoder(z_0)

        return recon_x, means, log_var, z_0, z_1, theta, logjcobin

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        # std = torch.sqrt(torch.exp(log_var))
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x



# add union test code for this py
if __name__ == '__main__':
    
    from thop import profile
    
    print('========Testing VAE========')
    # train(64, 512)
    vae = VAE(
        feature_dim = 1280,
        hidden_dim=256, 
        output_dim=12, 
        n_layers=4
    )
    vae = vae.cuda()
    print(vae)

    # get FLOPs and Parameters
    input_1 = torch.randn(64, 1280).cuda()
    input_2 = torch.randn(64, 1280).cuda()

    x, mean, var = vae.encoder(input_1)

    print("encoded X=", x.shape)

    flops, params = profile(vae, inputs=(input_1, input_2))
    flops = float(flops)
    params = float(params)

    print('FLOPs: {:.2f}M, Params: {:.2f}M'.format(flops/1e6, params/1e6))

    
