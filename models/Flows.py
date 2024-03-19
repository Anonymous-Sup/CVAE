import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import sys
sys.path.append('/home/zhengwei/Desktop/Zhengwei/Projects/CVAE/normalizing-flows')
from normflows.flows import Planar, Radial, MaskedAffineFlow, BatchNorm
from normflows import nets
# # from utils import idx2onehot
from functorch import vmap, jacfwd, grad
import torch.autograd.functional as F

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
                s = nets.MLP([input_dim, 8, input_dim], init_zeros=True)
                t = nets.MLP([input_dim, 8, input_dim], init_zeros=True)
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
    
class MLP(nn.Module):
    """A simple MLP with ReLU activations"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, leaky_relu_slope=0.2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

class YuKeMLPFLOW(nn.Module):

    def __init__(
            self,
            latent_size,
            hidden_dim=64,
            output_dim=1,
            num_layers=3
            ):
        super().__init__()

        self.latent_size = latent_size
        self.flows = nn.ModuleList([MLP(input_dim=hidden_dim+1, 
                                        hidden_dim=hidden_dim,
                                        output_dim=output_dim,  
                                        num_layers=num_layers) for _ in range(latent_size)])

        # self.fc = MLP(input_dim=embedding_dim, hidden_dim=hidden_dim,
        #               output_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x):
        
        # batch_size, latent_dim, hidden_dim+1
        # 64, 12, 64+1
        batch_size, latent_dim, feat_dim = x.shape

        sum_log_abs_det_jacobian = 0
        residuals = []
        
        for i in range(self.latent_size):
            # (batch_size, hidden_dim + x_dim)

            batch_inputs = x[:, i, :]
            
            residual = self.flows[i](batch_inputs)  # (batch_size, 1)
            # print("batch_inputs:{}".format(batch_inputs))
            # print("residual:{}".format(residual))
           

            J = jacfwd(self.flows[i])
            data_J = vmap(J)(batch_inputs).squeeze()

            # data_J = torch.where(torch.isnan(data_J), torch.zeros_like(data_J), data_J)
            logabsdet = torch.log(torch.abs(data_J[:, -1]) + 1e-8)

            # print("data_J:{}".format(data_J))
            print("logabsdet:{}".format(logabsdet))

            sum_log_abs_det_jacobian += logabsdet

            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, -1)

        return residuals, log_abs_det_jacobian
    
class YuKeMLPFLOW_onlyX_seperateZ(nn.Module):

    def __init__(
            self,
            latent_size=12, # 12, 36, 64
            hidden_dim=64,
            output_dim=1,
            num_layers=3
            ):
        super().__init__()

        self.latent_size = latent_size
        self.flows = nn.ModuleList([MLP(input_dim=1, 
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,  
                                num_layers=num_layers) for _ in range(latent_size)])
        # for flow in self.flows:
        #     for param in flow.parameters():
        #         if len(param.shape) > 1:  # weights
        #             nn.init.xavier_normal_(param)
        #         else:  # biases
        #             nn.init.zeros_(param)

    def forward(self, x):        
        # batch_size, latent_dim
        # 64, 12
        batch_size, latent_dim = x.shape

        sum_log_abs_det_jacobian = 0
        residuals = []
        
        for i in range(self.latent_size):

            # 64, 1
            batch_inputs = x[:, i]
            batch_inputs = batch_inputs.reshape(-1, 1)

            residual = self.flows[i](batch_inputs)  # (batch_size,1) --> (batch_size, 1)
            # print("residual:{}".format(residual))
            J = jacfwd(self.flows[i])
            
            
            # data_J.shape (64, 1, 1) --> (64)
            # batch, input dim, output dim?
            data_J = vmap(J)(batch_inputs).squeeze()

            # add for single dim input
            data_J = data_J.unsqueeze(1)

            logabsdet = torch.log(torch.abs(data_J[:, -1]))
            sum_log_abs_det_jacobian += logabsdet

            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        
        residuals = residuals.reshape(batch_size, -1)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, -1)

        return residuals, log_abs_det_jacobian
    



# above fuctions are abndon
class YuKeMLPFLOW_onlyX_seperateZ_init(nn.Module):

    def __init__(
            self,
            latent_size=12, # 12, 36, 64
            hidden_dim=64,
            output_dim=1,
            num_layers=3
            ):
        super().__init__()

        self.latent_size = latent_size
        self.flows = nn.ModuleList([MLP(input_dim=1, 
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,  
                                num_layers=num_layers) for _ in range(latent_size)])

        for flow in self.flows:
            for param in flow.parameters():
                if len(param.shape) > 1:  # weights
                    nn.init.xavier_normal_(param)
                else:  # biases
                    nn.init.zeros_(param)

    def forward(self, x):        
        # batch_size, latent_dim
        # 64, 12
        batch_size, latent_dim = x.shape

        sum_log_abs_det_jacobian = 0
        residuals = []
        
        for i in range(self.latent_size):

            # 64, 1
            batch_inputs = x[:, i].unsqueeze(-1)

            residual = self.flows[i](batch_inputs)  # (batch_size,1) --> (batch_size, 1)
            
            
            J = jacobian(self.flows[i], batch_inputs)
            # J.shape=torch.Size([64, 1, 64, 1])
            J_diagonal = J[:, 0, :, 0]  # Assuming J is a square matrix and we want the diagonal
            logabsdet_new = torch.log(torch.abs(J_diagonal) + 1e-6)  # small constant for stability

            # J = jacfwd(self.flows[i])
            # data_J = vmap(J)(batch_inputs).squeeze()
            # data_J = data_J.unsqueeze(1)
            # logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet_new

            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        
        residuals = residuals.reshape(batch_size, -1)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, -1)

        return residuals, log_abs_det_jacobian
class YuKeMLPFLOW_onlyX(nn.Module):

    def __init__(
            self,
            latent_size=12,
            hidden_dim=64,
            output_dim=12,
            num_layers=3
            ):
        super().__init__()

        self.latent_size = latent_size
        self.flows = MLP(input_dim=latent_size, 
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,  
                        num_layers=num_layers)

        # self.fc = MLP(input_dim=embedding_dim, hidden_dim=hidden_dim,
        #               output_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x):
        
        # batch_size, latent_dim, hidden_dim+1
        # 64, 12, 64+1
        batch_size, latent_dim = x.shape

        sum_log_abs_det_jacobian = 0
        
        # (batch_size, hidden_dim + x_dim)
        # 64, 12
        batch_inputs = x

        residual = self.flows(batch_inputs)  # (batch_size, 1)

        J = jacfwd(self.flows)

        # what is this shape?
        data_J = vmap(J)(batch_inputs).squeeze()

        logabsdet = torch.log(torch.abs(data_J[:, -1]))

        sum_log_abs_det_jacobian += logabsdet

        residual = residual.reshape(batch_size, -1)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, -1)

        return residual, log_abs_det_jacobian
    