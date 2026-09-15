import torch
import torch.nn as nn
from math import sqrt
from typing import Optional


class BaseAttention(nn.Module):
    """
    Base class for attention mechanisms. Users should extend this class and implement the compute_scores and combine_values methods.
    """

    def __init__(
        self,
        mask_flag: bool = True,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        masked_fill_value: float = float("-inf"),
    ):
        """
        Initialize the base attention mechanism.

        Parameters
        ----------
        mask_flag : bool, optional
            Whether to apply an attention mask, by default True
        scale : Optional[float], optional
            Scaling factor for the attention scores, by default None
        attention_dropout : float, optional
            Dropout rate for the attention weights, by default 0.1
        masked_fill_value : float, optional
            Value to use for masked positions in the attention scores, by default -inf. In case of numerical issues with -inf, consider using a large negative value like -1e9.

        """
        super(BaseAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.mask_flag = mask_flag
        self.masked_fill_value = masked_fill_value

    # -------- ABSTRACT METHODS -------- #
    def compute_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        pos_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Return attention scores of shape (B, H, L, S).

        Parameters
        ----------
        queries : torch.Tensor
            Query tensor of shape (B, L, H, E).
        keys : torch.Tensor
            Key tensor of shape (B, S, H, E).
        pos_bias :  torch.Tensor
            Positional bias tensor of shape (B, H, L, S)
        """
        raise NotImplementedError

    def combine_values(self, attn: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Return output of shape (B, L, H, D).

        Parameters
        ----------
        attn : torch.Tensor
            Attention weights of shape (B, H, L, S).
        values : torch.Tensor
            Value tensor of shape (B, S, H, D).
        """
        raise NotImplementedError

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_bias: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the forward pass of the attention mechanism.

        Parameters
        ----------
        queries : torch.Tensor
            Query tensor of shape (B, L, H, E).
        keys : torch.Tensor
            Key tensor of shape (B, S, H, E).
        values : torch.Tensor
            Value tensor of shape (B, S, H, D).
        attn_mask : torch.Tensor
            Attention mask tensor of shape (B, 1, L, S). Note that if the entire row is True, softmax will yield NaN. Consider using a key-only mask instead or filling masked positions with large negative values.
        pos_bias : torch.Tensor, optional
            Positional bias tensor of shape (B, H, L, S)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - Output tensor of shape (B, L, H, D).
            - Attention weights tensor of shape (B, H, L, S)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scale = (1.0 / sqrt(E)) if self.scale is None else self.scale

        scores = self.compute_scores(queries, keys, pos_bias)

        if self.mask_flag:
            attn_mask = attn_mask.to(dtype=torch.bool)
            masked_fill_value = self.masked_fill_value
            if masked_fill_value == float("-inf"):
                masked_fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attn_mask, masked_fill_value)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = self.combine_values(A, values)

        return V.contiguous(), A


class FullAttention(BaseAttention):
    """
    A full attention mechanism that computes attention scores using dot product and applies optional positional bias.
    """

    def compute_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention scores using dot product (blhe,bshe->bhls) and apply optional positional bias. Avoid using einsum for ONNX export compatibility.
        """
        queries = queries.permute(0, 2, 1, 3)  # B,H,L,E
        keys = keys.permute(0, 2, 3, 1)  # B,H,E,S
        scores = torch.matmul(queries, keys)  # B,H,L,S
        if pos_bias is not None:
            scores = scores + pos_bias
        return scores

    def combine_values(self, attn: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Combine attention weights with values to produce the output (bhls,bshd->blhd).
        """
        values = values.permute(0, 2, 1, 3)  # B,H,S,D
        out = torch.matmul(attn, values)  # B,H,L,D
        return out.permute(0, 2, 1, 3)  # B,L,H,D


class AttentionLayer(nn.Module):
    """
    Implementation of a multi-head attention layer.
    """

    def __init__(
        self,
        attention: BaseAttention,
        d_model: int,
        n_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ):
        """
        Initialize the multi-head attention layer.

        Parameters
        ----------
        attention : BaseAttention
            An instance of a BaseAttention subclass to perform the attention mechanism.
        d_model : int
            The dimensionality of the input and output feature vectors.
        n_heads : int
            The number of attention heads.
        d_keys : Optional[int], optional
            The dimensionality of the key and query vectors per head. If None, defaults to d_model // n_heads.
        d_values : Optional[int], optional
            The dimensionality of the value vectors per head. If None, defaults to d_model // n_heads.

        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the forward pass of the multi-head attention layer.
        """
        B, L = queries.size(0), queries.size(1)
        S = keys.size(1)
        H = self.n_heads

        queries = self.query_projection(queries).reshape(B, L, H, -1)
        keys = self.key_projection(keys).reshape(B, S, H, -1)
        values = self.value_projection(values).reshape(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            pos_bias=pos_bias,
        )
        out = out.reshape(B, L, -1)

        return self.out_projection(out), attn
