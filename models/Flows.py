import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import sys
sys.path.append('/home/zhengwei/Desktop/Zhengwei/Projects/CVAE/normalizing-flows')
from normflows.flows import Planar, Radial, MaskedAffineFlow, BatchNorm
from normflows import nets
# # from utils import idx2onehot


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
    def __init__(self, input_dim, flow_type=None, K=10):
        super().__init__()
        if flow_type== "Planar":
            flows = SimpleFlowModel([Planar((input_dim,), act="leaky_relu") for k in range(K)])
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
        else:
            raise ValueError("Invalid flow type: {}".format(flow_type))
        self.flows = flows

    def forward(self, x):
        theta, logjcobin = self.flows(x)
        return theta, logjcobin

    def inverse(self, theta):
        z, logjcobin = self.flows.inverse(theta)
        return z, logjcobin
    

class InvertibleMLPFlow(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, latent_dim),
            ) for _ in range(num_layers)
        ])
        
        # 初始化每一层的权重为正交矩阵
        # for layer in self.layers:
        #     nn.init.orthogonal_(layer[-1].weight)
        # 初始化每一层的权重为正交矩阵，并应用权重衰减
        for layer in self.layers:
            nn.init.orthogonal_(layer[-1].weight, gain=0.1)
            layer[-1].weight.register_hook(lambda grad: grad * 0.1)
        
    # def forward(self, z_u):
    #     log_det_jacobian = 0
    #     theta_u = z_u
    #     for layer in self.layers:
    #         theta_u = layer(theta_u)
    #         jacobian = torch.autograd.functional.jacobian(layer, theta_u)
    #         _, _, _, latent_u_dim = jacobian.shape
    #         jacobian = jacobian.view(-1, latent_u_dim, latent_u_dim)

    #         jacobian_float = jacobian.float()
    #         sign, logabsdet = torch.linalg.slogdet(jacobian_float)
    #         log_det_jacobian += sign.half() * logabsdet.half()
        
    #     log_det_jacobian = log_det_jacobian + 1e-8 # add small constant to prevent log(0)
    #     theta = theta_u[:, :self.latent_dim]
    #     return theta, log_det_jacobian
    
    def forward(self, z_u):
        log_det_jacobian = 0
        theta_u = z_u
        for layer in self.layers:
            theta_u = layer(theta_u)
            jacobian = torch.autograd.functional.jacobian(layer, theta_u)
            _, _, _, latent_u_dim = jacobian.shape
            jacobian = jacobian.view(-1, latent_u_dim, latent_u_dim)
            
            # Convert Jacobian to float32, compute log-det using logsumexp, and convert back to fp16
            jacobian_float = jacobian.float()
            log_det_jacobian += torch.logsumexp(torch.linalg.slogdet(jacobian_float)[1].half(), dim=0)
        
        theta = theta_u[:, :self.latent_dim]
        return theta, log_det_jacobian
    
    def inverse(self, theta_u):
        z_u = theta_u
        for layer in reversed(self.layers):
            with torch.no_grad():
                for sublayer in reversed(layer):
                    if isinstance(sublayer, nn.Linear):
                        z_u = torch.matmul(z_u - sublayer.bias, torch.pinverse(sublayer.weight))
                    else:
                        z_u = sublayer(z_u)
        z = z_u[:, :self.latent_dim]
        return z