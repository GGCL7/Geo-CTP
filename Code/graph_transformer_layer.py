import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads, use_bias=True):
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.out_dim = out_dim

        self.query_linear = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.key_linear = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.value_linear = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.edge_linear = nn.Linear(edge_dim, out_dim * num_heads, bias=use_bias)
        self.out_linear = nn.Linear(out_dim * num_heads, out_dim, bias=use_bias)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_attr):
        H = self.num_heads
        D = self.out_dim

        Q = self.query_linear(x).view(-1, H, D)
        K = self.key_linear(x).view(-1, H, D)
        V = self.value_linear(x).view(-1, H, D)

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        E = self.edge_linear(edge_attr).view(-1, H, D)

        scores = torch.einsum("bhd,bhd->bh", Q[edge_index[0]], K[edge_index[1]]) + torch.einsum("bhd,bhd->bh", E, E)
        scores = scores / (D ** 0.5)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        out = torch.einsum("bh,bhd->bhd", scores, V[edge_index[1]]).contiguous().view(-1, H * D)
        out = self.out_linear(out)


        out = scatter(out, edge_index[0], dim=0, reduce='mean')

        return out
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads, dropout=0.0, layer_norm=True, batch_norm=True):
        super(GraphTransformerLayer, self).__init__()
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim, edge_dim, num_heads, dropout)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

        if layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.attention(x, edge_index, edge_attr)

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h = self.ffn(h)

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)

        return h

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads, edge_dim, bert_feature_dim, dropout=0.0, layer_norm=True, batch_norm=True):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphTransformerLayer(in_channels, hidden_channels, edge_dim, num_heads, dropout, layer_norm, batch_norm))

        for _ in range(num_layers - 2):
            self.layers.append(GraphTransformerLayer(hidden_channels, hidden_channels, edge_dim, num_heads, dropout, layer_norm, batch_norm))

        self.layers.append(GraphTransformerLayer(hidden_channels, out_channels, edge_dim, num_heads, dropout, layer_norm, batch_norm))
        self.pool = global_mean_pool
        self.classifier1 = nn.Linear(out_channels, 320)
        self.classifier2 = nn.Linear(320 + bert_feature_dim, 2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        bert_features = data.bert_features

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        x = self.pool(x, batch)
        x1 = self.classifier1(x)
        x_combined = torch.cat((x1, bert_features), dim=1)
        output = self.classifier2(x_combined)

        return output, x1, bert_features




class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=0)
        sim = self.similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        N = z1.size(0)
        pos_sim = torch.diag(sim, N) + torch.diag(sim, -N)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0).reshape(2 * N, 1)

        neg_sim = sim.masked_select(~torch.eye(2 * N, dtype=bool).to(sim.device)).reshape(2 * N, -1)

        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * N).long().to(sim.device)

        loss = self.criterion(logits, labels)
        return loss / (2 * N)