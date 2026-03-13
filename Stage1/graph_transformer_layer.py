import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter, scatter_max, scatter_sum


def _ensure_batch(data, x):
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)
    return batch


def _segment_softmax(scores, source_index, num_heads):
    """Neighbor-wise softmax over edges that share the same source node."""
    num_edges = scores.size(0)
    head_ids = torch.arange(num_heads, device=scores.device).view(1, num_heads)
    segment_ids = source_index.view(-1, 1) * num_heads + head_ids

    flat_scores = scores.reshape(-1)
    flat_segments = segment_ids.reshape(-1)

    max_per_segment, _ = scatter_max(flat_scores, flat_segments, dim=0)
    stabilized = flat_scores - max_per_segment[flat_segments]
    exp_scores = stabilized.exp()
    denom = scatter_sum(exp_scores, flat_segments, dim=0)
    return (exp_scores / (denom[flat_segments] + 1e-12)).view(num_edges, num_heads)


class SequenceEncoderWrapper(nn.Module):
    """Accepts pooled [B, D] or residue-level [B, L, D] sequence embeddings."""

    def forward(self, sequence_features):
        if sequence_features.dim() == 1:
            return sequence_features.unsqueeze(0)
        if sequence_features.dim() == 2:
            return sequence_features
        if sequence_features.dim() == 3:
            return sequence_features.mean(dim=1)
        raise ValueError(f"Unsupported sequence feature shape: {tuple(sequence_features.shape)}")


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.t()) / self.temperature

        batch_size = z1.size(0)
        positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
        positives = positives.unsqueeze(1)

        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
        negatives = sim.masked_select(mask).view(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=sim.device)
        return self.criterion(logits, labels) / (2 * batch_size)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size, num_views, embedding_dim = features.shape
        features = F.normalize(features.view(batch_size * num_views, embedding_dim), dim=-1)
        expanded_labels = labels.unsqueeze(1).repeat(1, num_views).view(-1)

        logits = torch.matmul(features, features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        same_label = (expanded_labels.unsqueeze(0) == expanded_labels.unsqueeze(1)).float()
        same_label.fill_diagonal_(0.0)
        logits_mask = ~torch.eye(batch_size * num_views, dtype=torch.bool, device=features.device)

        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives_per_row = same_label.sum(dim=1)
        loss = -(same_label * log_prob).sum(dim=1) / positives_per_row.clamp_min(1.0)
        valid = positives_per_row > 0
        return loss[valid].mean() if valid.any() else loss.mean() * 0.0


class EdgeAwareMultiHeadGraphAttention(nn.Module):
    """
    Edge-aware graph attention that updates node messages using both node and edge states.

    Shapes:
    - node_states: [N, C]
    - edge_index: [2, E]
    - edge_states: [E, C]
    - attention logits/weights: [E, H]
    """

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.edge_score_proj = nn.Linear(hidden_dim, hidden_dim)
        self.node_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_out_proj = nn.Linear(num_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_states, edge_index, edge_states):
        src, dst = edge_index
        num_nodes = node_states.size(0)

        query = self.query_proj(node_states).view(num_nodes, self.num_heads, self.head_dim)
        key = self.key_proj(node_states).view(num_nodes, self.num_heads, self.head_dim)
        value = self.value_proj(node_states).view(num_nodes, self.num_heads, self.head_dim)

        edge_heads = edge_states.view(-1, self.num_heads, self.head_dim)
        edge_score_heads = self.edge_score_proj(edge_states).view(-1, self.num_heads, self.head_dim)

        scores = (
            (query[src] * key[dst]).sum(dim=-1)
            + (edge_score_heads * edge_heads).sum(dim=-1)
        ) / math.sqrt(self.head_dim)

        attention = _segment_softmax(scores, src, self.num_heads)
        attention = self.dropout(attention)

        # The paper term alpha_ij * V_ij * N_j is implemented as:
        # 1) project normalized neighbor node states with value_proj -> V_ij
        # 2) reshape the normalized neighbor state N_j into per-head chunks
        # 3) take an element-wise product V_ij * N_j, then weight by alpha_ij
        neighbor_state = node_states[dst].view(-1, self.num_heads, self.head_dim)
        neighbor_message = value[dst] * neighbor_state
        weighted_message = attention.unsqueeze(-1) * neighbor_message
        aggregated = scatter(weighted_message, src, dim=0, dim_size=num_nodes, reduce="sum")
        aggregated = aggregated.reshape(num_nodes, self.hidden_dim)

        node_update = self.node_out_proj(aggregated)
        edge_update = self.edge_out_proj(attention)
        return node_update, edge_update, attention


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0, layer_norm=True, batch_norm=False):
        super().__init__()
        self.attention = EdgeAwareMultiHeadGraphAttention(hidden_dim, num_heads, dropout)

        self.node_norm1 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.edge_norm1 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.node_norm2 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.edge_norm2 = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()

        self.node_batch_norm1 = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.edge_batch_norm1 = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.node_batch_norm2 = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.edge_batch_norm2 = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()

        self.node_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.edge_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_states, edge_index, edge_states):
        node_inputs = self.node_norm1(node_states)
        edge_inputs = self.edge_norm1(edge_states)

        node_attn_update, edge_attn_update, attention = self.attention(node_inputs, edge_index, edge_inputs)

        node_states = node_states + self.dropout(node_attn_update)
        edge_states = edge_states + self.dropout(edge_attn_update)
        node_states = self.node_batch_norm1(node_states)
        edge_states = self.edge_batch_norm1(edge_states)

        node_ffn_update = self.node_ffn(self.node_norm2(node_states))
        edge_ffn_update = self.edge_ffn(self.edge_norm2(edge_states))

        node_states = node_states + self.dropout(node_ffn_update)
        edge_states = edge_states + self.dropout(edge_ffn_update)
        node_states = self.node_batch_norm2(node_states)
        edge_states = self.edge_batch_norm2(edge_states)
        return node_states, edge_states, attention


class GraphTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, dropout=0.0, layer_norm=True, batch_norm=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_norm=layer_norm,
                    batch_norm=batch_norm,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, node_states, edge_index, edge_states):
        attentions = []
        for layer in self.layers:
            node_states, edge_states, attention = layer(node_states, edge_index, edge_states)
            attentions.append(attention)
        return node_states, edge_states, attentions


class GraphTransformer(nn.Module):
    """
    GeoCTP-style model with:
    - residue graph branch
    - edge-aware graph transformer encoder
    - pooled sequence branch
    - fusion classifier
    - projection heads for contrastive learning
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        num_heads,
        edge_dim,
        bert_dim,
        projection_dim,
        dropout=0.0,
        layer_norm=True,
        batch_norm=False,
        edge_hidden_dim=None,
        num_classes=2,
    ):
        super().__init__()
        edge_hidden_dim = edge_hidden_dim or hidden_channels
        if edge_hidden_dim != hidden_channels:
            raise ValueError("edge_hidden_dim must match hidden_channels in this implementation.")

        self.node_input_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_input_proj = nn.Linear(edge_dim, hidden_channels)
        self.encoder = GraphTransformerEncoder(
            hidden_dim=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
        )
        self.pool = global_mean_pool
        self.sequence_encoder = SequenceEncoderWrapper()

        self.structure_output = nn.Linear(hidden_channels, out_channels)
        self.fusion_classifier = nn.Linear(out_channels + bert_dim, num_classes)

        self.structure_proj = ProjectionHead(out_channels, projection_dim, dropout)
        self.sequence_proj = ProjectionHead(bert_dim, projection_dim, dropout)
        self.fused_proj = ProjectionHead(out_channels + bert_dim, projection_dim, dropout)

    def forward(self, data):
        node_states = self.node_input_proj(data.x)
        edge_attr = data.edge_attr if data.edge_attr.dim() == 2 else data.edge_attr.unsqueeze(-1)
        edge_states = self.edge_input_proj(edge_attr)
        batch = _ensure_batch(data, node_states)

        node_states, edge_states, attentions = self.encoder(node_states, data.edge_index, edge_states)
        pooled_structure = self.pool(node_states, batch)
        structure_embedding = self.structure_output(pooled_structure)

        sequence_features = self.sequence_encoder(data.bert_features)
        fused_embedding = torch.cat([structure_embedding, sequence_features], dim=-1)
        logits = self.fusion_classifier(fused_embedding)

        outputs = {
            "logits": logits,
            "structure_embedding": structure_embedding,
            "sequence_embedding": sequence_features,
            "structure_proj": self.structure_proj(structure_embedding),
            "sequence_proj": self.sequence_proj(sequence_features),
            "fused_embedding": fused_embedding,
            "fused_proj": self.fused_proj(fused_embedding),
            "node_states": node_states,
            "edge_states": edge_states,
            "attentions": attentions,
        }
        return outputs
