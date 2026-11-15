"""Hybrid GraphSAGE + 1D-CNN edge (transaction) classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class HybridGraphCNN(nn.Module):
    """GraphSAGE embeddings of sender, receiver and shared-entity nodes (card, device) + 1D-CNN over the sender's recent
    history + transaction features -> logit."""

    def __init__(self, node_dim, tx_dim, seq_ch, n_ent, hidden, gnn_out, cnn_ch, dropout, message_passing=True):
        super().__init__()
        self.mp = message_passing   # False = ablation: same inputs and sizes, but nodes never see their neighbours
        self.sage1 = SAGEConv(node_dim, hidden) if message_passing else nn.Linear(node_dim, hidden)
        self.sage2 = SAGEConv(hidden, gnn_out) if message_passing else nn.Linear(hidden, gnn_out)
        self.conv = nn.Conv1d(seq_ch, cnn_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(cnn_ch)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(nn.Linear((2 + n_ent) * gnn_out + cnn_ch + tx_dim, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))
        self.p = dropout

    def forward(self, x, edge_index, src, dst, ent, tx, seq):
        h = F.relu(self.sage1(x, edge_index) if self.mp else self.sage1(x))
        h = F.dropout(h, self.p, self.training)
        h = self.sage2(h, edge_index) if self.mp else self.sage2(h)    # [N, gnn_out]
        c = self.pool(F.relu(self.bn(self.conv(seq)))).squeeze(-1)      # [B, cnn_ch]
        ent_emb = h[ent].flatten(1) if ent.shape[1] else h.new_zeros(len(src), 0)   # card / device node embeddings
        return self.head(torch.cat([h[src], h[dst], ent_emb, c, tx], 1)).squeeze(-1)


def focal_loss(logits, target, alpha, gamma):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    a = alpha * target + (1 - alpha) * (1 - target)
    return (a * (1 - pt) ** gamma * bce).mean()
