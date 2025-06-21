import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from Bio import BiopythonWarning
import warnings


warnings.simplefilter('ignore', BiopythonWarning)


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=0)                           # [2B, D]
        sim = self.similarity(z.unsqueeze(1), z.unsqueeze(0))    # [2B, 2B]
        sim = sim / self.temperature

        B = z1.size(0)
        pos = torch.diag(sim, B) + torch.diag(sim, -B)            # [B] + [B]
        pos = torch.cat([pos, pos], dim=0).unsqueeze(1)           # [2B,1]
        neg = sim.masked_select(~torch.eye(2*B, device=sim.device).bool()) \
                 .view(2*B, -1)                                  # [2B, 2B-1]

        logits = torch.cat([pos, neg], dim=1)                    # [2B, 1+2B-1]
        labels = torch.zeros(2*B, dtype=torch.long, device=sim.device)
        loss = self.criterion(logits, labels)
        return loss / (2 * B)


class SupervisedContrastiveLoss(nn.Module):

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        B, V, D = features.shape
        feat = F.normalize(features, p=2, dim=2).view(B*V, D)    # [B*V, D]
        labs = labels.unsqueeze(1).repeat(1, V).view(-1)         # [B*V]
        sim_matrix = torch.matmul(feat, feat.T) / self.temperature
        mask = (labs.unsqueeze(0) == labs.unsqueeze(1)).float().to(feat.device)
        mask.fill_diagonal_(0)

        exp_sim = torch.exp(sim_matrix)
        numerator   = (exp_sim * mask).sum(dim=1)
        denominator = exp_sim.sum(dim=1) - torch.exp(torch.diagonal(sim_matrix))
        loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))
        return loss.mean()



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads, dropout, use_bias=True):
        super().__init__()
        self.H = num_heads
        self.D = out_dim
        self.q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.k = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.v = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.e = nn.Linear(edge_dim, out_dim * num_heads, bias=use_bias)
        self.out = nn.Linear(out_dim * num_heads, out_dim, bias=use_bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        Q = self.q(x).view(-1, self.H, self.D)
        K = self.k(x).view(-1, self.H, self.D)
        V = self.v(x).view(-1, self.H, self.D)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        E = self.e(edge_attr).view(-1, self.H, self.D)

        scores = (torch.einsum("bhd,bhd->bh", Q[edge_index[0]], K[edge_index[1]])
                + torch.einsum("bhd,bhd->bh", E, E)) / (self.D ** 0.5)
        alpha = F.softmax(scores, dim=-1)
        alpha = self.drop(alpha)
        out = torch.einsum("bh,bhd->bhd", alpha, V[edge_index[1]]).view(-1, self.H*self.D)
        out = self.out(out)
        return scatter(out, edge_index[0], dim=0, reduce='mean')


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads, dropout, layer_norm, batch_norm):
        super().__init__()
        self.attn = MultiHeadAttentionLayer(in_dim, out_dim, edge_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.bn1 = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim*2, out_dim)
        )
        self.ln2 = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        h = self.attn(x, edge_index, edge_attr)
        h = self.bn1(self.ln1(h))
        h = self.ffn(h)
        h = self.bn2(self.ln2(h))
        return h


class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, num_heads, edge_dim, bert_dim,
                 projection_dim, dropout, layer_norm, batch_norm):
        super().__init__()

        layers = []
        layers.append(GraphTransformerLayer(in_channels, hidden_channels, edge_dim,
                                            num_heads, dropout, layer_norm, batch_norm))
        for _ in range(num_layers - 2):
            layers.append(GraphTransformerLayer(hidden_channels, hidden_channels,
                                                edge_dim, num_heads, dropout,
                                                layer_norm, batch_norm))
        layers.append(GraphTransformerLayer(hidden_channels, out_channels, edge_dim,
                                            num_heads, dropout, layer_norm, batch_norm))
        self.layers = nn.ModuleList(layers)

        self.pool = global_mean_pool

        self.class1 = nn.Linear(out_channels, bert_dim)
        self.class2 = nn.Linear(bert_dim * 2, 2)

        self.proj_head = nn.Sequential(
            nn.Linear(bert_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        bert = data.bert_features
        if bert.dim() == 3:
            bert = bert.squeeze(1)


        for layer in self.layers:
            x = layer(x, ei, ea)
        x = self.pool(x, batch)

        h1 = self.class1(x)
        combo = torch.cat([h1, bert], dim=1)
        logits = self.class2(combo)
        z = self.proj_head(combo)
        return logits, h1, bert, z