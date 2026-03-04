import torch
from torch import nn
from torch_geometric.nn import GravNetConv
from typing import Optional
from base.model import BaseModel

from layers.attention import FullAttention, AttentionLayer
from layers.encoders import Encoder, VanillaEncoderLayer
from layers.embed import PositionalEmbedding
from utils.graph import reorder_from_graph_batches, pack_to_graph_batches


class ObjectCondensationAttn(BaseModel):
    """
    Object Condensation model for processing waveform and positional data to produce cluster positions in latent space and condensation strength. The architecture combines the following components.

    - Attention-based waveform encoder for temporal feature extraction per node
    - GravNet layers for geometric embedding which serves as positional encoding in attention
    - Multi-head self-attention encoder layers for relational learning among nodes
    - MLP heads for predicting latent cluster positions and condensation strength
    """

    def __init__(self, **kwargs):
        super(ObjectCondensationAttn, self).__init__()

        # Model hyperparameters
        # self.in_feats = kwargs.get('in_feats', 110)  # length of waveform
        self.pos_dim = kwargs.get('pos_dim', 2)  # dim of detector position (x,y)

        # lstm encoder params
        self.wf_d_model = kwargs.get('wf_d_model', 32)
        self.wf_enc_layers = kwargs.get('wf_enc_layers', 2)
        self.wf_enc_heads = kwargs.get('wf_enc_heads', 4)
        self.wf_enc_dropout = kwargs.get('wf_enc_dropout', 0.1)
        self.wf_enc_ff = kwargs.get('wf_enc_ff', self.wf_d_model * 4)

        # geometric embedding params
        self.n_gravnet_layers = kwargs.get('n_gravnet_layers', 2)
        self.gravnet_knn = kwargs.get('gravnet_knn', 5)

        # encoder parameters
        self.d_model = kwargs.get('d_model', 32)
        self.n_enc_layers = kwargs.get('n_enc_layers', 2)
        self.num_heads = kwargs.get('num_heads', 4)
        self.attn_dropout = kwargs.get('attn_dropout', 0.1)
        self.attn_ff = kwargs.get('attn_ff', self.d_model * 4)

        # oc mlp parameters
        self.oc_mlp_pos_hidden = kwargs.get('oc_mlp_pos_hidden', 64)
        self.oc_mlp_dropout = kwargs.get('oc_mlp_dropout', 0.1)
        self.oc_mlp_beta_hidden = kwargs.get('oc_mlp_beta_hidden', 64)

        ################################################################################
        # Attentional layers for waveform processing
        ################################################################################
        self.wf_embedding = nn.Linear(1, self.wf_d_model)
        self.wf_pe = PositionalEmbedding(d_model=self.wf_d_model)
        wf_attn_layers = [
            VanillaEncoderLayer(
                attn_layer=AttentionLayer(
                    FullAttention(
                        mask_flag=False,
                        attention_dropout=self.wf_enc_dropout,
                        scale=None,
                    ),
                    d_model=self.wf_d_model,
                    n_heads=self.wf_enc_heads,
                ),
                d_model=self.wf_d_model,
                ff_kwargs={
                    "d_ff": self.wf_enc_ff,
                    "activation": nn.LeakyReLU,
                },
                dropout=self.wf_enc_dropout,
                batchnorm=False,
            )
            for _ in range(self.wf_enc_layers)
        ]

        self.wf_enc = Encoder(wf_attn_layers)
        self.wf_proj = nn.Conv1d(
            in_channels=self.wf_d_model,
            out_channels=self.d_model,
            kernel_size=3,
            padding=1,
        )

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
            nn.Linear(self.oc_mlp_pos_hidden, self.pos_dim),
        )

        self.oc_mlp_beta = nn.Sequential(
            nn.Linear(self.d_model, self.oc_mlp_beta_hidden),
            nn.ReLU(),
            nn.Linear(self.oc_mlp_beta_hidden, 1),  # output: beta
            nn.Sigmoid(),  # ensure beta is in [0, 1]
        )

        # # initialize the bias of the final beta layer to a negative value (sigmoid(-3) ≈ 0.047).
        # with torch.no_grad():
        #     self.oc_mlp_beta[-2].bias.fill_(-3.0)
        #     self.oc_mlp_pos[-1].weight.fill_(0.0)
        #     self.oc_mlp_pos[-1].bias.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Compute the forward pass of the Object Condensation model with attention.

        Parameters
        ----------
        x: torch.Tensor
            Input waveform features, shape [N, L], where N is number of nodes, L is waveform length.
        pos: torch.Tensor
            Input positional features, shape [N, pos_dim], where pos_dim is typically 2
        batch: torch.Tensor, optional
            Batch vector assigning each node to a graph in the batch, shape [N]. If None, all nodes are assumed to belong to a single graph.
        mask: torch.Tensor, optional
            Boolean mask tensor indicating valid nodes, shape [N]. If None, all nodes are considered valid.

        Returns
        -------
        x_c: torch.Tensor
            Predicted cluster positions in latent space, shape [N, pos_dim].
        beta: torch.Tensor
            Predicted condensation strength for each node, shape [N, 1].
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if mask is None:
            mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        """
        Encode input waveforms with attention-based encoder
        """
        x = x[mask]
        x = x.unsqueeze(-1)  # (N, L) -> (N, L, 1)
        x = self.wf_embedding(x)  # (N, L, 1) -> (N, L, wf_d_model)
        x = self.wf_pe(x)
        x, _ = self.wf_enc(x)

        x = x.permute(0, 2, 1)  # (N, L, wf_d_model) -> (N, wf_d_model, L)
        x = self.wf_proj(x)  # (N, wf_d_model, L) -> (N, d_model, L)
        x = x.mean(
            dim=-1
        )  # global average pooling over waveform length -> (N, d_model)

        """
        Use GravNet to embed geometric positional information
        - input: original position tensor [N, pos_dim]
        - output: [N, d_model]
        """
        pos = self.geo_mlp(pos)  # [N, d_model]
        for gl in range(self.n_gravnet_layers):
            pos = self.geo_embed[gl](pos, batch=batch)  # [N, d_model]
        pos = pos[mask]

        """
        Attentional encoder for graph processing
        - pack to graph-batched format, record
            - idx_out: for restoring original node order
            - attn_mask: (B, N_max) where [b, :N_b] = True
        - pass through attention encoder
        """
        x, pos, idx_out, valid = pack_to_graph_batches(x, pos, batch=batch[mask])

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

        x, _ = self.attn_enc(x, pos_bias=None, attn_mask=attn_mask)
        # discard padded nodes and restore original order
        x = reorder_from_graph_batches(x, idx_out)

        """
        clustering here to compute latent-space x_c and condensation strength \beta
        """
        x_c = self.oc_mlp_pos(x)
        beta = self.oc_mlp_beta(x)

        return x_c, beta
