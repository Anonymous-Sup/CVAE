import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, number_layers=3, leak_relu_slope=0.2):
        super(MLP, self).__init__()

        layers = []
        for l in range(number_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leak_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leak_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.MLP = nn.Sequential(*layers)

    def forward(self, x):
        x = self.MLP(x)
        return x

class NIPS(nn.Module):
    def __init__(self, VAE, FLOWs, feature_dim, hidden_dim, latent_size):
        super(NIPS, self).__init__()

        self.VAE = VAE
        self.FLOWs = FLOWs

        self.latent_size = latent_size

        if hidden_dim is None:
            self.hidden_dim = 768 # default hidden_dim 12 * 64
            print("hidden_dim is not specified, set to be {}".format(self.hidden_dim))
        else:
            self.hidden_dim = hidden_dim

        self.domain_embeding = MLP(feature_dim, self.hidden_dim, self.hidden_dim, number_layers=3)
        
        # self.fusion = MLP(latent_size+self.hidden_dim, self.hidden_dim, latent_size, number_layers=3)

        # self.domain_embeding.apply(self.init_weights)
        # self.fusion.apply(self.init_weights)

    def forward(self, x, domain_index):
        
        domain_index_feat= self.domain_embeding(domain_index)
        domain_index_feat = self.normalization(domain_index_feat)

        # (64,12)
        x_proj, means, log_var = self.VAE.encoder(x)

        # 64, 12
        z_0 = self.VAE.reparameterize(means, log_var)

        # 64, 12, 64
        domian_dim =  int(self.hidden_dim/self.latent_size)
        U = domain_index_feat.view(-1, self.latent_size, domian_dim)
        # (64, 12, 1) + (64, 12, 64) -> (64, 12, 65)
        flow_input = torch.cat((x_proj.unsqueeze(-1), U), dim=-1)
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
        z_1 = flow_input

        theta, logjcobin = self.FLOWs(z_1)

        recon_x = self.VAE.decoder(x_proj)

        return recon_x, means, log_var, z_0, x_proj, z_1, theta, logjcobin, domain_index_feat, flow_input
    
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