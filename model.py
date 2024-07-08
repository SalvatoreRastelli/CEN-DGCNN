import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *



class CEN_DGCNN(nn.Module):
    def __init__(self, input_dim=None, channel_dim=None, edgef_dim=None, layers=None, device='cpu', dropout=0.5):
        super(CEN_DGCNN, self).__init__()

        self.device=device

        self.layers = nn.ModuleList()
    
        self.layers.append(CENDGCNN_layer(input_dim, channel_dim, edgef_dim, num_layer=1 , device=device))

        for i in range(1, layers):
            self.layers.append(CENDGCNN_layer((channel_dim*edgef_dim), channel_dim, edgef_dim, num_layer=i+1 , device=device))
        
        self.dropout = dropout
        self.padding = channel_dim*edgef_dim
        

    def forward(self, node_features, edge_features, edge_list):
        """
        node_features: torch.Tensor of shape [N, F]
            N is the number of nodes
            F is the number of node features
            
        edge_features: torch.Tensor of shape [N, N, P]
            N is the number of nodes
            P is the number of edge features

        edge_list: torch.Tensor of shape [2,E]
            E in the number of edges
        
        Returns:
            x: torch.Tensor of shape [N, out_dim]
            e: torch.Tensor of shape [N, N, P]
        """
        
        # CEN_DGCNN--> ReLU --> Dropout
        N = node_features.shape[0]
        f = node_features.shape[1]
        zeros = torch.zeros(N,f, device=self.device)

        #First Layer
        x , e = self.layers[0](node_features , edge_features , node_features , zeros , edge_list)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        
        x_l1 = list()
        x_l1.append(x)
        x_0 = torch.matmul(node_features,torch.eye(f,self.padding, device=self.device))

        #Second Layer
        x , e = self.layers[1](x , edge_features , x_0 , x_0 , edge_list)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_l1.append(x)

        for layer in self.layers[2:]:
            x , e = layer(x, e, x_0, x_l1[0], edge_list)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_l1.append(x)
            x_l1.pop(0)
            
        
        return x , e  

    
