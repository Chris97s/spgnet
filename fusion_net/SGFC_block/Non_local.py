import torch
from torch import nn
from torch.nn import functional as F


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


class NLattention(nn.Module):
    def __init__(self,channel,num_points,d_model):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLattention, self).__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, num_points, d_model))
        self.ln = nn.LayerNorm(channel)
    def forward(self, x):
        """
        args
            x:  b,sequence,c
        """
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation

        Q = x   #b,s,c
        K_T = x.permute(0, 2, 1) #b,c,s
        V = x   #b,s,c
        attention = torch.matmul(Q, K_T) #b,s,s
        N = attention.size(-1) 
        attention_softmax = F.softmax(attention, dim=-1) #b,s,s
        attention_softmax = attention_softmax / N
        y = torch.matmul(attention_softmax, V) #b,s,c
        # contiguous here just allocates contiguous chunk of memory
        y = self.ln(y+x)
        return y
    
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear layers for queries, keys, and values for each head
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)

        # Linear layer for the output of multi-head attention
        self.out_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()

        # Linearly project queries, keys, and values for each head
        queries = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Compute scaled dot-product attention for each head
        attention = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / (self.head_dim**0.5)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to values
        out = torch.matmul(attention, values).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, input_dim)
        
        # Linearly project the concatenated multi-head output
        out = self.out_linear(out)

        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_self_attention = MultiHeadSelfAttention(input_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Multi-Head Self-Attention
        attn_output = self.multihead_self_attention(x)
        # Residual Connection and Layer Normalization
        x = self.norm1(x + attn_output)
        # Feedforward Layer
        ff_output = self.feedforward(x)
        # Residual Connection and Layer Normalization
        x = self.norm2(x + ff_output)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers,num_points):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(input_dim, num_heads) for _ in range(num_layers)])
        self.position_embedding = nn.Parameter(torch.randn(1, num_points, input_dim))
    def forward(self, x):
        x = x + self.position_embedding
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':

    tansformer = TransformerEncoder(32,8,1,4)
    input = torch.randn(1,4,32)
    output = tansformer(input)




#     import torch

#     for bn_layer in [True, False]:
#         # img = torch.zeros(2, 3, 20)
#         # net = NLBlockND(in_channels=3, mode='concatenate', dimension=1, bn_layer=bn_layer)
#         # out = net(img)
#         # print(out.size())

#         img = torch.zeros(2, 3, 20, 20)
#         net = NLBlockND(in_channels=3, mode='concatenate', dimension=2, bn_layer=bn_layer)
#         out = net(img)
#         print(out.size())

#         # img = torch.randn(2, 3, 8, 20, 20)
#         # net = NLBlockND(in_channels=3, mode='concatenate', dimension=3, bn_layer=bn_layer)
#         # out = net(img)
#         # print(out.size())

