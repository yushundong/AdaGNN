import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Adagnn_without_weight(Module):

    def __init__(self, diag_dimension, in_features, out_features, bias=True):
        super(Adagnn_without_weight, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diag_dimension = diag_dimension
        self.learnable_diag_1 = Parameter(torch.FloatTensor(in_features))  # in_features

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.learnable_diag_1, mean=0, std=0)

    def forward(self, input, l_sym):

        e1 = torch.spmm(l_sym, input)
        alpha = torch.diag(self.learnable_diag_1)
        e2 = torch.mm(e1, alpha)
        e4 = torch.sub(input, e2)
        output = e4

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Adagnn_with_weight(Module):

    def __init__(self, diag_dimension, in_features, out_features, bias=True):
        super(Adagnn_with_weight, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diag_dimension = diag_dimension
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.learnable_diag_1 = Parameter(torch.FloatTensor(in_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.normal_(self.learnable_diag_1, mean=0, std=0.01)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, l_sym):
        e1 = torch.spmm(l_sym, input)
        alpha = torch.diag(self.learnable_diag_1)
        e2 = torch.mm(e1, alpha + torch.eye(self.in_features, self.in_features).cuda())
        e4 = torch.sub(input, e2)
        e5 = torch.mm(e4, self.weight)
        output = e5

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
