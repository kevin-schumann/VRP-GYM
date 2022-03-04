import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np


class GraphDecoder(nn.Module):
    """
    Decoder Class to generate node prediction.
    """

    def __init__(
        self,
        emb_dim: int = 128,
        num_heads: int = 8,
        v_dim: int = 128,
        k_dim: int = 128,
    ):
        """
        Args:
            emb_dim (int): Dimension of the input embedding.
            num_heads (int): Number of attention heads.
            v_dim (int): Dimension of attention value.
            k_dim (int): Dimension of attention key.
        """
        super().__init__()

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

        # project in context of [graph_emb, ]
        self._context_proj = nn.Linear(emb_dim * 2 + 1, emb_dim * 3, bias=False)

        self.first_ = None
        self.last_ = None
        self.first_step = True
        self.num_heads = num_heads

    def forward(
        self,
        node_embs: torch.Tensor,
        mask: torch.Tensor = None,
        load: torch.Tensor = None,
        C: int = 10,
        rollout: bool = False,
    ):
        """
        Forward method of the Decoder

        Args:
            node_embs (torch.Tensor): Node embeddings with shape (batch_size, num_nodes, emb_dim)
            mask (torch.Tensor, optional): Node mask with shape (batch_size, num_nodes). Defaults to None.
            load (torch.Tensor, optional): Load of the vehicle with shape (batch_size, 1). Defaults to None.
            C (int, optional): Hyperparameter to regularize logit calculation. Defaults to 10.
            rollout (bool, optional): Determines if prediction is sampled or maxed. Defaults to False.

        Returns:
            torch.Tensor: Node prediction for each graph with shape (batch_size, 1)
            torch.Tensor: Log probabilities
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
        if load is None:
            context = torch.cat([graph_emb, self.first_, self.last_], -1)
        else:
            context = torch.cat([graph_emb, self.last_, load[:, None, None]], -1)
            context = self._context_proj(context)

        attn_mask = mask.repeat(self.num_heads, 1).unsqueeze(1)
        q, _ = self.attention(context, node_embs, node_embs, attn_mask=attn_mask)
        q = self._att_output(q)

        u = torch.tanh(q.bmm(k.transpose(-2, -1)) / emb_dim ** 0.5) * C
        u = u.masked_fill(mask.unsqueeze(1).bool(), float("-inf"))

        log_prob = torch.zeros(size=(batch_size,))
        nn_idx = None
        if rollout:
            nn_idx = u.argmax(-1)
        else:
            m = Categorical(logits=u)
            nn_idx = m.sample()
            log_prob = m.log_prob(nn_idx)
        temp = nn_idx.unsqueeze(-1).repeat(1, 1, emb_dim)
        self.last_ = torch.gather(node_embs, 1, temp)

        if self.first_step:
            self.first_ = self.last_
            self.first_step = False

        return nn_idx, log_prob

    def reset(self):
        """
        Resets first and last node.
        Must be called before starting a new game.
        """
        self.first_ = None
        self.last_ = None
        self.first_step = True

