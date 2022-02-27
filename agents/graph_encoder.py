from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.module):
    def __init__(self, embedding_dim, hidden_dim, num_attention_layers, num_heads):
        depot_dim: int = 2
        node_dim: int = 2

        # initial embeds ff layer for each nodes type
        self.depot_embed = nn.Linear(depot_dim, embedding_dim)
        self.node_embed = nn.Linear(node_dim, embedding_dim)

        self.attention_layers = [
            MultiHeadAttentionLayer(
                embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads
            )
            for _ in num_attention_layers
        ]

    def forward(self, x):
        """
        Returns the embedding for each node

        Args:
            x (Tuple): Returns tuple of x and the embeddings.
        """

        out = torch.concat(
            # TODO embed nodes and depots differently.
        )

        for i in range(self.attention_layers):
            out = self.attention_layers[i](out)

        return (out, torch.mean(out, dim=1))


class MultiHeadAttentionLayer(nn.module):
    def __init__(self, embedding_dim, hidden_dim, num_heads):
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads
        )

        self.batch_norm1 = nn.BatchNorm1d(num_features=embedding_dim)
        self.batch_norm2 = nn.BatchNorm1d(num_features=hidden_dim)

        self.ln1 = nn.Linear(embedding_dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        residual = x

        # calc attention, skip connection and normalize
        attention = self.multi_head_attention(x, x, x)
        out = torch.add(residual, attention)
        residual = self.batch_norm1(out)

        # pass through fully connected layers
        out = self.ln1(residual)
        out = F.relu(out)
        out = self.ln2(out)

        # add residual and renormalized
        out = torch.add(residual, out)
        out = self.batch_norm2(out)

        return out

