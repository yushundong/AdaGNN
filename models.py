import torch.nn as nn
import torch.nn.functional as F
from adagnn.layers import Adagnn_without_weight, Adagnn_with_weight


class AdaGNN(nn.Module):
    def __init__(self, diag_dimension, nfeat, nhid, nlayer, nclass, dropout):
        super(AdaGNN, self).__init__()

        self.should_train_1 = Adagnn_with_weight(diag_dimension, nfeat, nhid)
        assert nlayer - 2 >= 0
        self.hidden_layers = nn.ModuleList([
            Adagnn_without_weight(nfeat, nhid, nhid, bias=False)
            for i in range(nlayer - 2)
        ])
        self.should_train_2 = Adagnn_with_weight(diag_dimension, nhid, nclass)
        self.dropout = dropout

    def forward(self, x, l_sym):


        x = F.relu(self.should_train_1(x, l_sym))  # .relu
        x = F.dropout(x, self.dropout, training=self.training)

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x, l_sym)
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.should_train_2(x, l_sym)  # + res1
        return F.log_softmax(x, dim=1)


