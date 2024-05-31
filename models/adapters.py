from __future__ import division
import torch
import torch.nn as nn


"""Sparsemax activation function.
Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
Code from: https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py
"""
class Sparsemax(nn.Module):
    def __init__(self, dim=None):
        """Initialize sparsemax activation.
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim


    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device="cuda", dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output


    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class SparseBattery(nn.Module):
    def __init__(self, num_adapters, c_in, c_out, usebias=True):
        # num_adapters = 128, c_in = 1280, c_out = 64
        super(SparseBattery, self).__init__()
        self.gate = nn.Sequential(nn.Linear(c_in, num_adapters, bias=usebias), Sparsemax(dim=1))
        self.adapters = nn.ModuleList([nn.Linear(c_in, c_out, bias=usebias) for _ in range(num_adapters)])

    def forward(self, x):
        # Contract batch over height and width
        # this g is the pre-activation q, q with shape (128,4) (batchsize, num_adapters)
        g = self.gate(x)

        h = []
        for k in range(len(self.adapters)):
            h.append(g[:, k].view(-1, 1) * self.adapters[k](x))
            '''
            suppose k = 0 
            g[:, 0].view(-1, 1) = (batchsize, 1)
            g[:, 0].view(-1, 1) * self.adapters[0](x) = (batchsize, c_out)
            shape of self.adapters[0](x) = (batchsize, c_out)
            '''
        out = sum(h)
        # each element in h has the same shape with out = (batchsize, c_out)
        return g, out


class SparseLatentAdapter(nn.Module):
    def __init__(self, num_adapters, c_in, c_out, use_bais=True):
        super(SparseLatentAdapter, self).__init__()
        
        self.linear = nn.Linear(c_in, c_out, bias=use_bais)
        self.bn = nn.BatchNorm1d(c_out)
        self.parallel_adapter = SparseBattery(num_adapters, c_in, c_out, use_bais)

    def forward(self, x):
        # this is f_0(x), which is a shared linear
        y = self.linear(x)
    
        # this is the sum of K adapters with gate and indendepnt 1*1 linear
        # gate.shape = (batchsize, num_adapters)
        # out.shape = (batchsize, 64)
        gate, out = self.parallel_adapter(x)

        # this f_0(x) + sum_k(g_k(x) * f_k(x))
        y = y + out
        y = self.bn(y)

        return gate, y