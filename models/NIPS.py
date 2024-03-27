import torch
import torch.nn as nn

def idx2onehot(idx, n):
    # (64, 25)
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, number_layers=3, leak_relu_slope=0.2):
        super(MLP, self).__init__()

        layers = []
        for l in range(number_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
                layers.append(nn.LeakyReLU(leak_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                layers.append(nn.LeakyReLU(leak_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        layers.append(nn.LeakyReLU(leak_relu_slope))
        self.MLP = nn.Sequential(*layers)

    def forward(self, x):
        x = self.MLP(x)
        return x

class NIPS(nn.Module):
    def __init__(self, VAE, FLOWs, feature_dim, hidden_dim, latent_size, only_x=False, use_centroid=False):
        super(NIPS, self).__init__()

        self.VAE = VAE
        self.FLOWs = FLOWs

        self.latent_size = latent_size
        self.only_x_input = only_x
        self.use_centroid = use_centroid

        if hidden_dim is None:
            self.hidden_dim = 768 # default hidden_dim 12 * 64
            print("hidden_dim is not specified, set to be {}".format(self.hidden_dim))
        else:
            self.hidden_dim = hidden_dim

        if self.use_centroid:
            self.domain_embeding = MLP(feature_dim, self.hidden_dim, self.hidden_dim, number_layers=4)
        else:
            # 50 --> 768 --> 768
            self.domain_embeding = MLP(25*2, self.hidden_dim, self.hidden_dim, number_layers=4)
        # self.fusion = MLP(latent_size+self.hidden_dim, self.hidden_dim, latent_size, number_layers=3)

    def forward(self, x, domain_index):
        
        if self.use_centroid:
            assert len(domain_index.size()) == 2
        else:
            assert len(domain_index.size()) == 1
            domain_index = idx2onehot(domain_index, 50)

        domain_feature = self.domain_embeding(domain_index)
        domain_feature = domain_feature
        # print("domain_feature", domain_feature)
        if torch.isnan(domain_feature).any():
            print("domain_feature has nan")
        # domain_index_feat = self.normalization(domain_feature)
        # domain_index_feat = domain_feature / domain_feature.norm(dim=1, keepdim=True)
        domain_index_feat = self.normalize_l2(domain_feature)
        if torch.isnan(domain_index_feat).any():
            print("domain_feature after normalization has nan")

        # (64,12)
        x_proj, means, log_var = self.VAE.encoder(x)

        # x_proj_norm = self.normalization(x_proj)
        x_proj_norm = self.normalize_l2(x_proj)
        # x_proj_norm = x_proj

        # 64, 12
        z_0 = self.VAE.reparameterize(means, log_var)

        # 64, 12, 64
        domian_dim =  int(self.hidden_dim/self.latent_size)
        U = domain_index_feat.view(-1, self.latent_size, domian_dim)
        # (64, 12, 1) + (64, 12, 64) -> (64, 12, 65)
        flow_input = torch.cat((U, x_proj_norm.unsqueeze(-1)), dim=-1)
        # print("flow_input", flow_input)
        # (64, 12, 65) -> (64, 65*12)
        # flow_input = flow_input.view(-1, self.latent_size * (1 + self.hidden_dim/self.latent_size))

        # flow_input = torch.cat((z_0, domain_index_feat), dim=-1)
        # z_1 = self.fusion(flow_input)
        '''
        Ablation test Mar17 20:23
        z_1 = x_proj
        test_only_x_input = True

        origin: z_1 = flow_input
        '''
        if self.only_x_input:
            z_1 = x_proj_norm
        else:
            z_1 = flow_input

        theta, logjcobin = self.FLOWs(z_1)

        recon_x = self.VAE.decoder(x_proj_norm)

        return recon_x, means, log_var, z_0, x_proj, x_proj_norm, z_1, theta, logjcobin, domain_index_feat, flow_input
    

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