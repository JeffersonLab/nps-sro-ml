import torch
from torch import nn
from torch_geometric.nn import GravNetConv
from typing import Optional
from base.model import BaseModel

from layers.attention import FullAttention, AttentionLayer
from layers.encoders import Encoder, VanillaEncoderLayer
from utils.graph import reorder_from_graph_batches, pack_to_graph_batches


def unpack_features(
    x: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function for disentangling energy and time features, creating a mask for valid pulses. Assumes input x has shape [N, L] where L is even and represents (E, t) pairs for pulses. Returns reshaped features and mask.

    Parameters
    ----------
    x : torch.Tensor
        Input features, shape [N, L], where N is number of nodes, L is max feature length (e.g. max number of pulses per node times 2).
    batch : torch.Tensor, optional
        Batch vector assigning each node to a graph in the batch, shape [N]. If None, all nodes are assumed to belong to a single graph.

    Returns
    -------
    x_out : torch.Tensor
        Shape [N, P, 2], where P = L // 2.
    mask : torch.Tensor
        Shape [N, P], True where pulse is non-zero.
    """
    assert x.dim() == 2, "Input x should be of shape [N, L]"
    assert x.size(1) % 2 == 0, "Feature dimension must be even (E, t pairs)"

    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

    N, L = x.size()
    P = L // 2

    x_out = x.view(N, P, 2)

    mask = x_out.abs().sum(dim=-1) > 0  # (N, P)
    return x_out, mask


class PulseSetEncoder(nn.Module):
    def __init__(self, hidden=64, out_dim=32):
        super().__init__()

        # pulse-level embedding φ
        self.phi = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # block-level compression ρ
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

        self.ln = nn.LayerNorm(out_dim)

    def forward(self, pulses, mask):
        """
        pulses: [N_blocks, K_max, 2]  (E, t)
        mask:   [N_blocks, K_max]     (True for valid pulse)
        """
        x = self.phi(pulses)  # [N, K, hidden]

        x = x * mask.unsqueeze(-1)  # zero out padded pulses

        x = x.sum(dim=1)  # permutation invariant aggregation
        x = self.rho(x)  # [N, out_dim]
        x = self.ln(x)
        return x


class ObjectCondensationAttnForSimData(BaseModel):

    def __init__(self, **kwargs):
        super(ObjectCondensationAttnForSimData, self).__init__()

        # Model hyperparameters
        self.pos_dim = kwargs.get('pos_dim', 2)  # dim of detector position (x,y)

        self.embed_in = kwargs.get('embed_in', 2)
        self.embed_out = kwargs.get('embed_out', 32)

        # geometric embedding params
        self.n_gravnet_layers = kwargs.get('n_gravnet_layers', 2)
        self.gravnet_knn = kwargs.get('gravnet_knn', 5)

        # encoder parameters
        self.d_model = kwargs.get('d_model', 64)
        self.n_enc_layers = kwargs.get('n_enc_layers', 2)
        self.num_heads = kwargs.get('num_heads', 4)
        self.attn_dropout = kwargs.get('attn_dropout', 0.1)
        self.attn_ff = kwargs.get('attn_ff', self.d_model * 4)

        # oc mlp parameters
        self.oc_mlp_pos_hidden = kwargs.get('oc_mlp_pos_hidden', 64)
        self.oc_mlp_pos_dim = kwargs.get('oc_mlp_pos_dim', 8)
        self.oc_mlp_dropout = kwargs.get('oc_mlp_dropout', 0.1)
        self.oc_mlp_beta_hidden = kwargs.get('oc_mlp_beta_hidden', 64)

        ################################################################################
        # embed input features with MLP
        ################################################################################
        self.set_encoder = PulseSetEncoder(hidden=self.embed_in, out_dim=self.embed_out)

        ################################################################################
        # Embed geometric information
        ################################################################################
        self.geo_mlp = nn.Sequential(
            nn.Linear(self.pos_dim, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        gravnet_layers = [
            GravNetConv(
                in_channels=self.d_model,
                out_channels=self.d_model,
                space_dimensions=self.pos_dim,
                propagate_dimensions=self.d_model,
                k=self.gravnet_knn,
            )
            for _ in range(self.n_gravnet_layers)
        ]

        self.geo_embed = nn.ModuleList(gravnet_layers)

        ################################################################################
        # Attentional encoder layers for graph processing
        ################################################################################
        self.attn_embedding = nn.Linear(self.embed_out, self.d_model)

        attn_layers = [
            VanillaEncoderLayer(
                attn_layer=AttentionLayer(
                    FullAttention(
                        mask_flag=True,
                        attention_dropout=self.attn_dropout,
                        scale=None,
                    ),
                    d_model=self.d_model,
                    n_heads=self.num_heads,
                ),
                d_model=self.d_model,
                ff_kwargs={
                    "d_ff": self.attn_ff,
                    "activation": nn.LeakyReLU,
                },
                dropout=self.attn_dropout,
                batchnorm=False,
            )
            for _ in range(self.n_enc_layers)
        ]

        self.attn_enc = Encoder(attn_layers)

        self.oc_mlp_pos = nn.Sequential(
            nn.Linear(self.d_model, self.oc_mlp_pos_hidden),
            nn.ReLU(),
            nn.Dropout(self.oc_mlp_dropout),
            nn.Linear(self.oc_mlp_pos_hidden, self.oc_mlp_pos_hidden),
            nn.ReLU(),
            nn.Linear(self.oc_mlp_pos_hidden, self.oc_mlp_pos_dim),
        )

        self.oc_mlp_beta = nn.Sequential(
            nn.Linear(self.d_model, self.oc_mlp_beta_hidden),
            nn.ReLU(),
            nn.Linear(self.oc_mlp_beta_hidden, 1),  # output: beta
            nn.Sigmoid(),  # ensure beta is in [0, 1]
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):
        """
        Compute the forward pass of the Object Condensation model with attention.

        Parameters
        ----------
        x: torch.Tensor
            Input features, shape [N, P, 2], where N is number of nodes, P is max number of pulses, 2 corresponds to (E, t) for each pulse. Padded pulses should be masked out.
        pos: torch.Tensor
            Input positional features, shape [N, pos_dim], where pos_dim is typically 2
        batch: torch.Tensor, optional
            Batch vector assigning each node to a graph in the batch, shape [N]. If None, all nodes are assumed to belong to a single graph.

        Returns
        -------
        x_c: torch.Tensor
            Predicted cluster positions in latent space, shape [N, pos_dim].
        beta: torch.Tensor
            Predicted condensation strength for each node, shape [N, 1].
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x, mask = unpack_features(x, batch=batch)
        x = self.set_encoder(x, mask)

        """
        Use GravNet to embed geometric positional information
        - input: original position tensor [N, pos_dim]
        - output: [N, d_model]
        """
        pos = self.geo_mlp(pos)  # [N, d_model]
        for gl in range(self.n_gravnet_layers):
            pos = self.geo_embed[gl](pos, batch=batch)  # [N, d_model]

        """
        Attentional encoder for graph processing
        - pack to graph-batched format, record
            - idx_out: for restoring original node order
            - attn_mask: (B, N_max) where [b, :N_b] = True
        - compute geo positional bias
        - pass through attention encoder
        """
        x, pos, idx_out, valid = pack_to_graph_batches(x, pos, batch=batch)
        x = self.attn_embedding(x)  # [B, N_max, d_model]

        # add geo positional embedding
        x = x + pos  # [B, N_max, d_model]

        """
        if scores of (padded Q x padded K) are masked to -inf, softmax will yield NaN
        instead, we create attn mask based on key only -> scores in (padded Q x valid K)
        it is the standard approach as padded nodes should be discarded downstream
        see my gist
        """
        key_mask = ~valid[:, None, :]  # [B, 1, N_max]
        attn_mask = key_mask.unsqueeze(1)  # [B, 1, 1, N_max]

        x, attns = self.attn_enc(x, pos_bias=None, attn_mask=attn_mask)

        if torch.isnan(x).any():
            print("NaN detected in attention output!")
            print(f"Valid nodes per graph: {valid.sum(dim=1)}")
            print(f"Attention mask shape: {attn_mask.shape}")
            print(f"Any all-masked rows: {attn_mask.all(dim=-1).any()}")
            for i, attn in enumerate(attns):
                if attn is not None and torch.isnan(attn).any():
                    print(f"Layer {i} attention has NaN")

        # discard padded nodes and restore original order
        x = reorder_from_graph_batches(x, idx_out)

        """
        clustering here to compute latent-space x_c and condensation strength \beta
        """
        x_c = self.oc_mlp_pos(x)
        beta = self.oc_mlp_beta(x)

        return x_c, beta
