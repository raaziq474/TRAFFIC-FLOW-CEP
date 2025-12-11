import torch
import torch.nn as nn
import torch.nn.functional as F_func
import math


class DynamicGraphConstructor(nn.Module):
    def __init__(self, num_nodes, hidden_dim, k=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.k = k
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_dim))

    def forward(self, device=None):
        embeddings = self.node_embeddings
        if device is not None:
            embeddings = embeddings.to(device)

        norm_embeddings = F_func.normalize(embeddings, p=2, dim=1)
        adj = torch.mm(norm_embeddings, norm_embeddings.t())

        if self.k < self.num_nodes:
            top_k_values, top_k_indices = torch.topk(adj, self.k, dim=1)
            mask = torch.zeros_like(adj)
            mask.scatter_(1, top_k_indices, 1)
            adj = adj * mask

        adj = adj + torch.eye(self.num_nodes, device=adj.device)

        rowsum = adj.sum(1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat = torch.diag(d_inv)
        adj = torch.mm(d_mat, adj)

        return adj


class DiffusionGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(K + 1, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        B, N, F_in = x.shape

        supports = [x]
        x_k = x
        for _ in range(self.K):
            x_k = torch.matmul(adj, x_k)
            supports.append(x_k)

        supports = torch.stack(supports, dim=0)
        output = torch.einsum('kbnf,kfo->bno', supports, self.weight)

        return output + self.bias


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, N, F = x.shape
        x = x.reshape(B * N, T, F)

        Q = self.q_proj(x).reshape(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).reshape(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F_func.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).reshape(B * N, T, F)

        output = self.out_proj(attn_output)
        return output.reshape(B, T, N, F)


class GatedTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv_gate = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=(padding, 0))
        self.conv_filter = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=(padding, 0))
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        gate = torch.sigmoid(self.conv_gate(x))
        filt = torch.tanh(self.conv_filter(x))
        out = gate * filt

        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)

        return out


class D2STGNNBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, K=2, kernel_size=3):
        super().__init__()

        self.spatial_conv = DiffusionGraphConv(hidden_dim, hidden_dim, K=K)
        self.spatial_norm = nn.LayerNorm(hidden_dim)

        self.temporal_attn = TemporalSelfAttention(hidden_dim, num_heads)
        self.temporal_conv = GatedTemporalConv(hidden_dim, hidden_dim, kernel_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, adj):
        B, F, T, N = x.shape

        # Spatial
        x_spatial = x.permute(0, 2, 3, 1).reshape(B * T, N, F)
        x_spatial = self.spatial_conv(x_spatial, adj)
        x_spatial = x_spatial.reshape(B, T, N, F)
        x_spatial = self.spatial_norm(x_spatial)
        x_spatial = x_spatial.permute(0, 3, 1, 2)
        x = x + self.dropout(x_spatial)

        # Temporal Attention
        x_attn = x.permute(0, 2, 3, 1)
        x_attn = self.temporal_attn(x_attn)
        x_attn = x_attn.permute(0, 3, 1, 2)
        x = x + self.dropout(x_attn)

        # Temporal Conv
        x_conv = self.temporal_conv(x)
        x = x + self.dropout(x_conv)

        return x


class D2STGNN(nn.Module):
    def __init__(self, num_nodes, num_classes, num_layers=4, hidden_dim=64,
                 num_heads=4, K=3, k_neighbors=10):
        super().__init__()

        self.num_nodes = num_nodes
        self.output_sequence_length = num_classes
        self.hidden_dim = hidden_dim

        self.graph_constructor = DynamicGraphConstructor(num_nodes, hidden_dim, k=k_neighbors)
        self.input_proj = nn.Linear(1, hidden_dim)

        self.classifier = nn.Sequential(
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(self.hidden_dim, num_classes)
        )

        self.blocks = nn.ModuleList([
            D2STGNNBlock(hidden_dim, num_heads=num_heads, K=K)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        # project T_in -> T_out
        self.temp_conv_T_proj = None

    def forward(self, x):
        # x: (B, T_in, N)
        B, T_in, N = x.shape
        device = x.device

        # Init Conv1D if shape changes
        if self.temp_conv_T_proj is None or self.temp_conv_T_proj.in_channels != T_in:
            self.temp_conv_T_proj = nn.Conv1d(
                in_channels=T_in,
                out_channels=self.output_sequence_length,
                kernel_size=1
            ).to(device)

        # Graph
        adj = self.graph_constructor(device=device)

        # Input embedding
        x = x.unsqueeze(-1)
        x = self.input_proj(x)        # (B, T_in, N, F)
        x = x.permute(0, 3, 1, 2)     # (B, F, T_in, N)

        # Blocks
        for block in self.blocks:
            x = block(x, adj)

        # Map F -> 1
        x = x.permute(0, 2, 3, 1)                # (B, T_in, N, F)

        # Use only last timestep for classification
        x = x[:, -1, :, :]                      # (B, N, F)

        # Fully connected classifier
        x = self.classifier(x)                  # (B, N, num_classes)

        return x

