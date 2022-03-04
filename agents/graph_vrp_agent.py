from typing import Tuple

import torch

from agents.graph_encoder import GraphDemandEncoder

from .graph_tsp_agent import TSPAgent, TSPModel


class VRPModel(TSPModel):
    def __init__(
        self,
        depot_dim: int,
        node_dim: int,
        emb_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        num_heads: int,
    ):
        """
        The VRPModel is used in companionship with the VRPEnv
        to solve the capacited vehicle routing problem.

        Args:
            depot_dim (int): Input dimension of a depot node.
            node_dim (int): Input dimension of a regular graph node.
            emb_dim (int): Size of a vector in the embedding space.
            hidden_dim (int): Dimension of the hidden layers of the 
                ff-network layers within the graph-encoder.
            num_attention_layers (int): Number of attention layers 
                for both the graph-encoder and -decoder.
            num_heads (int): Number of attention heads in each 
                MultiHeadAttentionLayer for both the graph-encoder and -decoder.
        """
        super().__init__(
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        )

        self.encoder = GraphDemandEncoder(
            depot_input_dim=depot_dim,
            node_input_dim=node_dim,
            embedding_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        )

    def forward(self, env, rollout=False) -> Tuple[float, float]:
        """
        Forward method of the model
        Args:
            env (gym.Env): environment which the agent has to solve.
            rollout (bool, optional): policy decision. Defaults to False.

        Returns:
            Tuple[float, float]: accumulated loss and log probabilities.
        """
        done = False
        state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)
        acc_loss = torch.zeros(size=(state.shape[0],), device=self.device)
        acc_log_prob = torch.zeros(size=(state.shape[0],), device=self.device)

        emb = self.encoder(x=state[:, :, :2], depot_mask=state[:, :, 3].bool())

        while not done:
            actions, log_prob = self.decoder(
                node_embs=emb, mask=state[:, :, -1], rollout=rollout,
            )

            state, loss, done, _ = env.step(actions.cpu().numpy())

            acc_loss += torch.tensor(loss, dtype=torch.float, device=self.device)
            acc_log_prob += log_prob.squeeze().to(self.device)

            state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

        self.decoder.reset()

        return acc_loss, acc_log_prob  # shape (batch_size), shape (batch_size)


class VRPAgent(TSPAgent):
    def __init__(
        self,
        depot_dim: int = 2,
        node_dim: int = 2,
        emb_dim: int = 128,
        hidden_dim: int = 512,
        num_attention_layers: int = 3,
        num_heads: int = 8,
        lr: float = 1e-4,
        csv_path: str = "loss_log.csv",
        seed=69,
    ):
        """
        The VPRAgent is used in companionship with the VPREnv
        to solve the vehicle routing problem.

        Args:
            depot_dim (int): Input dimension of a graph depot.
            node_dim (int): Input dimension of a regular graph node.
            emb_dim (int): Size of a vector in the embedding space.
            hidden_dim (int): Dimension of the hidden layers of the 
                ff-network layers within the graph-encoder.
            num_attention_layers (int): Number of attention layers 
                for both the graph-encoder and -decoder.
            num_heads (int): Number of attention heads in each 
                MultiHeadAttentionLayer for both the graph-encoder and -decoder.
            lr (float): learning rate.
            csv_path (string): file where the loss gets saved.
            seed (int): the seed.
        """
        super().__init__(
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
            lr=lr,
            csv_path=csv_path,
            seed=seed,
        )
        self.model = VRPModel(
            depot_dim=depot_dim,
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        ).to(self.device)

        self.target_model = VRPModel(
            depot_dim=depot_dim,
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        ).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
