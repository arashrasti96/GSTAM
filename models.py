import os.path as osp
from math import ceil
import torch
from torch_geometric.nn import DenseSAGEConv, DenseGCNConv, GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import LayerNorm, InstanceNorm


class DenseGCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nconvs=3, dropout=0, if_mlp=False, net_norm='none', pooling='mean', **kwargs):
        super(DenseGCN, self).__init__()

        self.molhiv = False
        if kwargs['args'].dataset in ['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace']:
            nclass = 1
            self.molhiv = True

        if nconvs == 1:
            nhid = nclass

        self.mlp = if_mlp
        if self.mlp:
            DenseGCNConv = nn.Linear
        else:
            from torch_geometric.nn import DenseSAGEConv, DenseGCNConv
        self.convs = nn.ModuleList([])
        self.convs.append(DenseGCNConv(nfeat, nhid))
        for _ in range(nconvs-1):
            self.convs.append(DenseGCNConv(nhid, nhid))

        self.norms = nn.ModuleList([])

        for _ in range(nconvs):
            if nconvs == 1:  norm = torch.nn.Identity()
            elif net_norm == 'none':
                norm = torch.nn.Identity()
            elif net_norm == 'batchnorm':
                norm = BatchNorm1d(nhid)
            elif net_norm == 'layernorm':
                norm = nn.LayerNorm([nhid], elementwise_affine=True)
            elif net_norm == 'instancenorm':
                norm = InstanceNorm(nhid, affine=False) #pyg
            elif net_norm == 'groupnorm':
                norm = nn.GroupNorm(4, nhid, affine=True)
            self.norms.append(norm)

        self.lin3 = torch.nn.Linear(nhid, nclass) if nconvs != 1 else lambda x: x
        self.dropout = dropout
        self.pooling = pooling

    def forward(self, x, adj, mask=None, if_embed=False, if_attention = False, dist=False,p=1):
        if self.dropout !=0:
            x_mask = torch.distributions.bernoulli.Bernoulli(self.dropout).sample([x.size(0), x.size(1)]).to('cuda').unsqueeze(-1)
            x = x_mask * x
         ######################
        attention_outs = []
        Distri = []
        ######################
        for i in range(len(self.convs)):
            if self.mlp:
                x = self.convs[i](x)
            else:
                x = self.convs[i](x, adj, mask)
            x = self.perform_norm(i, x)
            x = F.relu(x)
            ###################################################
            #if if_attention:
             #   squared_tensor = x**p
              #  sq = squared_tensor.sum(dim=0)
               # sq = sq.sum(dim=0)
         
             #   #sq = F.normalize(sq[0,...],p=2)
              #  sq = sq/torch.norm(sq,1)
              #  attention_outs.append(sq)
               # # x = global_add_pool(x, batch=data.batch)
        #if if_attention:
         #   return attention_outs
            if if_attention:
                squared_tensor = x**p
                sq = squared_tensor.sum(dim=0)
                sq = torch.matmul(sq.t(),sq)
                sq = sq/torch.norm(sq,1)
                attention_outs.append(sq)
        if if_attention:
            return attention_outs
            ####################################################
        if self.pooling == 'sum':
            x = x.sum(1)
        if self.pooling == 'mean':
            x = x.mean(1)
        if if_embed:
            return x
        if dist:
            sq = x
            sq = sq.sum(dim=0)
            sq = sq/torch.norm(sq,1)
            return sq
        if self.molhiv:
            x = self.lin3(x)
        else:
            x = self.lin3(x)

            x = F.log_softmax(x, dim=-1)

        return x


    def embed(self, x, adj, mask=None):
        return self.forward(x, adj, mask, if_embed=True)
    ###########################
    def attention(self, x, adj, mask=None, power=1):
        p = power
        return self.forward(x, adj, mask, if_attention=True,p = p)
    def distribution(self, x, adj, mask=None):
        return self.forward(x, adj, mask, dist = True)
    ############################
    def perform_norm(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = self.norms[i](x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x
        
        
#######################################################################################################
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

class GIN(torch.nn.Module):
    def __init__(self, nfeat, nconvs, nhid, nclass, net_norm, pooling, dropout, args):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(torch.nn.Sequential(
            torch.nn.Linear(nfeat, nhid),
            torch.nn.ReLU(),
            torch.nn.Linear(nhid, nhid),
            torch.nn.BatchNorm1d(nhid)
        )))
        for _ in range(nconvs - 1):
            self.convs.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(nhid, nhid),
                torch.nn.ReLU(),
                torch.nn.Linear(nhid, nhid),
                torch.nn.BatchNorm1d(nhid)
            )))
        self.fc = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.pooling = pooling

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Example usage
#model = GIN(nfeat=dataset.num_features, nconvs=nconv, nhid=args.hidden, nclass=dataset.num_classes, net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, args=args)
#######################################################################################################
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, nfeat, nconvs, nhid, nclass, net_norm, pooling, dropout, args, heads=1):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(nfeat, nhid, heads=heads))
        for _ in range(nconvs - 1):
            self.convs.append(GATConv(nhid * heads, nhid, heads=heads))
        self.fc = torch.nn.Linear(nhid * heads, nclass)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Example usage
#model = GAT(nfeat=dataset.num_features, nconvs=nconv, nhid=args.hidden, nclass=dataset.num_classes, net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, args=args, heads=8)

######################################################################################################
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MPNN(MessagePassing):
    def __init__(self, nfeat, nconvs, nhid, nclass, net_norm, pooling, dropout, args):
        super(MPNN, self).__init__(aggr='add')
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(nconvs - 1):
            self.convs.append(torch.nn.Linear(nhid, nhid))
        self.fc = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.pooling = pooling

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        for conv in self.convs:
            x = self.propagate(edge_index, x=x)
            x = F.relu(conv(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.pooling == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def message(self, x_j, edge_weight=None):
        return x_j

    def update(self, aggr_out):
        return aggr_out

# Example usage
#model = MPNN(nfeat=dataset.num_features, nconvs=nconv, nhid=args.hidden, nclass=dataset.num_classes, net_norm=args.net_norm, pooling=args.pooling, dropout=args.dropout, args=args)
