import torch
import torch.nn as nn
from math import sqrt
from typing import Optional
from utils.masks import TriangularCausalMask, MaskProtocol


class BaseAttention(nn.Module):
    """
    Base class for attention mechanisms. Users should extend this class and implement the compute_scores and combine_values methods.
    """

    def __init__(
        self,
        *,
        mask_flag: bool = True,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
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
        output_attention : bool, optional
            Whether to return the attention weights along with the output, by default False
        masked_fill_value : float, optional
            Value to use for masked positions in the attention scores, by default -inf. In case of numerical issues with -inf, consider using a large negative value like -1e9.

        """
        super(BaseAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.mask_flag = mask_flag
        self.masked_fill_value = masked_fill_value

    # -------- ABSTRACT METHODS -------- #
    def compute_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return attention scores of shape (B, H, L, S).

        Parameters
        ----------
        queries : torch.Tensor
            Query tensor of shape (B, L, H, E).
        keys : torch.Tensor
            Key tensor of shape (B, S, H, E).
        pos_bias : Optional[torch.Tensor], optional
            Positional bias tensor of shape (B, H, L, S), by default None
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
        *,
        pos_bias: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor | MaskProtocol] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        pos_bias : Optional[torch.Tensor], optional
            Positional bias tensor of shape (B, H, L, S), by default None
        attn_mask : Optional[torch.Tensor | MaskProtocol], optional
            Attention mask tensor or protocol, by default None. If provided, should be broadcastable to shape (B, 1, L, S). Note that if the entire row is True, softmax will yield NaN. Consider using a key-only mask instead or filling masked positions with large negative values.

        Returns
        -------
        tuple[torch.Tensor, Optional[torch.Tensor]]
            A tuple containing:
            - Output tensor of shape (B, L, H, D).
            - Attention weights tensor of shape (B, H, L, S) if output_attention is True, else None.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = self.compute_scores(queries, keys, pos_bias)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device).mask
            else:
                if hasattr(attn_mask, "mask"):
                    attn_mask = attn_mask.mask

                attn_mask = attn_mask.to(dtype=torch.bool)

            scores.masked_fill_(attn_mask, self.masked_fill_value)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = self.combine_values(A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


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
        Compute attention scores using dot product and apply optional positional bias.
        """
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if pos_bias is not None:
            scores = scores + pos_bias
        return scores

    def combine_values(self, attn: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Combine attention weights with values to produce the output.
        """
        return torch.einsum("bhls,bshd->blhd", attn, values)


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
        *,
        attn_mask: Optional[torch.Tensor] = None,
        pos_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform the forward pass of the multi-head attention layer.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        if pos_bias is not None:
            pos_bias = pos_bias.unsqueeze(1).expand(B, H, L, S)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            pos_bias=pos_bias,
            attn_mask=attn_mask,
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
