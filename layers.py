import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def g_l(x , W, l,out_dim, lamb = 0.5, device="cpu"):
    f = x.shape[1]
    return (1 - delta_l(l,lamb))* torch.matmul(x, torch.eye(f , out_dim, device=device)) + delta_l(l,lamb) * torch.matmul(x,W)

def delta_l(l , lamb):
    return math.log((lamb/l) + 1)


def DSN(E):
    """
    :param E: Edge embedding of size [N, N, P].
    :return E_norm: edge matrix normalized.
    """

    E = E / (torch.sum(E, dim=1, keepdim=True) + 1e-16 ) # normalised across rows
    F = E / (torch.sum(E, dim=0, keepdim=True) + 1e-16 )  # normalised across cols
    E_norm = torch.einsum('ijp,kjp->ikp', E, F)

    return E_norm





class GCNLayer(nn.Module):
    def __init__(self, input_dim=None, channel_dim=None , edgef_dim=None , bias=True , num_layer=None , device='cpu'):
        super(GCNLayer, self).__init__() 

        self.layer = num_layer
        self.channel_dim = channel_dim
        self.output_dim = edgef_dim*channel_dim 
        self.device = device

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, channel_dim))

        self.zeta = nn.Parameter(torch.FloatTensor(1))
        self.eta = nn.Parameter(torch.FloatTensor(1))
        self.theta = nn.Parameter(torch.FloatTensor(1))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

        nn.init.uniform_(self.zeta)
        nn.init.uniform_(self.eta)
        nn.init.uniform_(self.theta)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


    def forward(self, X, E, X_0, X_l1, attention):
        """
        param X: node features [N, node_features]
        param E: Edge filter [N, N, edge_features]
        param X_0: node features at layer(0)
        param X_l1: node features at layer(l-1)
        returns output: updated node features embedding
        """

        support0 = g_l(X, self.weight, self.layer, self.channel_dim, device=self.device)
        support1 = g_l(X_l1, self.weight, self.layer, self.channel_dim, device=self.device)
        support2 = g_l(X_0, self.weight, self.layer, self.channel_dim, device=self.device)

        
        channels = E.shape[2] #number of edge features
        N = support0.size(0) #number of nodes
        features = support0.size(1) # number of nodes features
        out_tensor = torch.zeros((N, features, channels), device=self.device) 
        

        for channel in range(channels):
            support = torch.matmul(attention*E[:,:, channel], support0)

            out_tensor[:,:,channel] = self.zeta*support + self.eta * support1 + self.theta*support2


        # concatenate the output tensor along the channel dimension
        out_shape = (-1, self.output_dim)
        output = out_tensor.transpose(2, 1).reshape(out_shape)

        
        if self.bias is not None:
            return output + self.bias
        else:
            return output





class AttentionFilter(nn.Module):
    """
    Attention Module to make the edge update and filter.
    """
    def __init__(self, in_features, device='cpu'):
        super(AttentionFilter, self).__init__()
        self.dropout = 0.2
        self.in_features = in_features

        self.device=device

        self.out_features=64

        self.W = nn.Parameter(torch.empty(size=(in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, X, adj):
        Wh = torch.mm(X, self.W) 
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        return attention
        

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)




class CENDGCNN_layer(nn.Module):
    def __init__(self, input_dim=None, channel_dim=None , edgef_dim=None, num_layer=None , device='cpu'):
        super(CENDGCNN_layer, self).__init__()

        self.device=device

        self.GCN = GCNLayer(input_dim, channel_dim, edgef_dim, num_layer=num_layer, device=device)
        self.filter = AttentionFilter(input_dim , device=device)


    def forward(self, X, E, X_0, X_l1, edge_list):
        """
        X, X_0, X_l1: torch.Tensor of shape [N, F]
            N is the number of nodes
            F is the number of node features
            
        E: torch.Tensor of shape [N, N, P]
            P is the number of features
            
        edge_list: torch.Tensor of shape [2,E]
            E in the number of edges
        
        Returns:
           X_upd: torch.Tensor of shape [N, F]
           E_upd: torch.Tensor of shape [N, N, P]
        """
        adjacency = torch.zeros(( X.size(0) , X.size(0) ), device=self.device)
        for i,j in edge_list.t():
            adjacency[i,j] = adjacency[j,i] = 1

        #filter
        update = self.filter(X, adjacency)
    
        #Node embedding update
        X_upd = self.GCN(X, E, X_0, X_l1, update)

        #Edge embedding update
        E_upd = DSN(torch.mul(update.unsqueeze(-1).repeat(1, 1, 3), E))

        return  X_upd, E_upd