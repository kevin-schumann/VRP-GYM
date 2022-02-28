import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphDecoder(nn.Module):
    def __init__(self, emb_dim=128, num_heads=8, v_dim=2, k_dim=2, seed=69):
        super().__init__()
        torch.manual_seed(seed)

        self._first_node = nn.Parameter(torch.rand(1, 1, emb_dim))
        self._last_node = nn.Parameter(torch.rand(1, 1, emb_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=3 * emb_dim,
            num_heads=num_heads,
            kdim=k_dim,
            vdim=v_dim,
            batch_first=True,
        )

        self._kp = nn.Linear(emb_dim, emb_dim, bias=False)
        self._att_output = nn.Linear(emb_dim * 3, emb_dim, bias=False)

        self.first_ = None
        self.last_ = None

    def forward(
        self, node_embs: torch.Tensor, mask: torch.Tensor = None, C=10, rollout=False
    ):
        """
        Forward method for the decoder.
        *Add desc_*

        Args:
            node_emb (np.ndarray): node embeddings of shape. Shape: (batch_size, num_nodes, emb_dim)
        """
        batch_size, _, emb_dim = node_embs.shape

        graph_emb = torch.mean(
            node_embs, axis=1, keepdims=True
        )  # shape (batch, 1, emb)

        if self.first_ is None:
            self.first_ = self._first_node.repeat(batch_size, 1, 1)
            self.last_ = self._last_node.repeat(batch_size, 1, 1)

        k = self._kp(node_embs)

        # Create context with first, last node and graph embedding.
        # Where last is the node from last decoding step.
        context = torch.cat([graph_emb, self.first_, self.last_], -1)

        q, _ = self.attention(context, node_embs, node_embs)
        q = self._att_output(q)

        u = torch.tanh(q.bmm(k.transpose(-2, -1)) / emb_dim ** 0.5) * C
        u = u.masked_fill(mask.unsqueeze(1), float("-inf"))

        if rollout:
            nn_idx = u.argmax(-1)
        else:
            # sampling
            ...
        temp = nn_idx.unsqueeze(-1).repeat(1, 1, emb_dim)
        self.last_ = torch.gather(node_embs, 1, temp)

        if len(mask[mask == 1]) == 0:
            self.first_ = self.last_
        return nn_idx

    def reset(self):
        self.first_ = None
        self.last_ = None
