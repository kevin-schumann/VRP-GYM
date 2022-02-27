import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphDecoder(nn.module):
    def __init__(self, hidden_dim, num_attention_layers, num_heads):
        super.__init__()

    def forward(
        self, node_emb: np.ndarray, last_node_ids: np.ndarray, mask: np.ndarray = None
    ):
        """
        Forward method for the decoder.
        *Add desc_*

        Args:
            node_emb (np.ndarray): node embeddings of shape. Shape: (batch_size, num_nodes, emb_dim)
        """
        # TODO: Test
        # Calculate graph emb
        graph_emb = np.mean(node_emb, axis=1, keepdims=True)  # shape (batch, 1, emb)

        # Create context last node and graph embedding.
        # Where last is the node from last decoding step.
        # The paper also mentioned to use the first node, but cant find code for that rn.
        context = (
            graph_emb + node_emb[:, last_node_ids, :]
        )  # TODO: need to handle demand here
