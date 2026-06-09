import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """Original GCNII layer."""

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GraphConvolutionDS(nn.Module):
    """One DSGCN propagation layer.

    Implements
        H^(l+1) = sigma((1-alpha) * (eta_l * S H^(l) + (1-eta_l) * H^(l) W_l) + alpha * H^(0)).

    eta_l follows a bounded inverse-polynomial decay schedule.
    """

    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolutionDS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, alpha, eta_l):
        hi = torch.spmm(adj, input)
        transformed = torch.mm(input, self.weight)
        output = (1 - alpha) * (eta_l * hi + (1 - eta_l) * transformed) + alpha * h0
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)


class DSGCN(nn.Module):
    """DSGCN model built on top of the GCNII project skeleton."""

    def __init__(
        self,
        nfeat,
        nlayers,
        nhidden,
        nclass,
        dropout,
        alpha,
        eta0=0.8,
        eta_min=0.1,
        eta_lambda=4.0,
        eta_power=1.5,
        residual=False,
    ):
        super(DSGCN, self).__init__()
        self.nlayers = nlayers
        self.dropout = dropout
        self.alpha = alpha
        self.eta0 = eta0
        self.eta_min = eta_min
        self.eta_lambda = eta_lambda
        self.eta_power = eta_power
        self.act_fn = nn.ReLU()

        self.convs = nn.ModuleList(
            [GraphConvolutionDS(nhidden, nhidden, residual=residual) for _ in range(nlayers)]
        )
        self.fcs = nn.ModuleList([nn.Linear(nfeat, nhidden), nn.Linear(nhidden, nclass)])
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())

        eta_schedule = self._build_eta_schedule(nlayers, eta0, eta_min, eta_lambda, eta_power)
        self.register_buffer("eta_schedule", torch.tensor(eta_schedule, dtype=torch.float32))

    @staticmethod
    def _build_eta_schedule(nlayers: int, eta0: float, eta_min: float, eta_lambda: float, eta_power: float) -> List[float]:
        if nlayers <= 0:
            return []
        if nlayers == 1:
            return [float(eta0)]
        schedule = []
        for layer_idx in range(nlayers):
            progress = layer_idx / float(nlayers - 1)
            eta_l = eta_min + (eta0 - eta_min) * (1.0 + eta_lambda * progress) ** (-eta_power)
            schedule.append(float(eta_l))
        return schedule

    def get_eta_schedule(self) -> List[float]:
        return [float(v) for v in self.eta_schedule.detach().cpu().tolist()]

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        h0 = self.act_fn(self.fcs[0](x))
        h = h0
        for i, con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            eta_l = self.eta_schedule[i]
            h = self.act_fn(con(h, adj, h0, self.alpha, eta_l))
        h = F.dropout(h, self.dropout, training=self.training)
        logits = self.fcs[-1](h)
        return F.log_softmax(logits, dim=1)


class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner


if __name__ == '__main__':
    pass
