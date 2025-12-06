import torch
import torch.nn as nn
from layers.attention import AttentionLayer


class BaseEncoderLayer(nn.Module):
    """
    Base class for an encoder layer in a transformer model, which encapsulates attention, feedforward mechanisms, and normalization.
    """

    def __init__(
        self,
        attn_layer: AttentionLayer,
        d_model: int = 512,
        dropout: float = 0.1,
        batchnorm: bool = False,
        return_attn: bool = True,
        ff_kwargs: dict = {},
    ):
        """
        Initialize the base encoder layer.

        Parameters
        ----------
        attn_layer : AttentionLayer
            The attention layer to be used in the encoder.
        d_model : int, optional
            The dimension of the model, by default 512.
        dropout : float, optional
            Dropout rate, by default 0.1.
        batchnorm : bool, optional
            Whether to use batch normalization, by default False.
        return_attn : bool, optional
            Whether to return attention weights, by default True.
        ff_kwargs : dict, optional
            Additional keyword arguments for the feedforward network, by default {}.
        """
        super(BaseEncoderLayer, self).__init__()
        self.attn_layer = attn_layer
        self.feedforward = self._build_feedforward(d_model, **ff_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = batchnorm

        if batchnorm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.return_attn = return_attn

    def _build_feedforward(self, d_model: int, **ff_kwargs) -> nn.Module:
        """
        Build the feedforward network for the encoder layer. User must override this method.

        Parameters
        ----------
        d_model : int
            The dimension of the model.
        ff_kwargs : dict
            Additional keyword arguments for the feedforward network.

        Returns
        -------
        nn.Module
            The feedforward network module.
        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        pos_bias: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass through the encoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the encoder layer, typically of shape (batch_size, sequence_length, d_model).
        pos_bias : torch.Tensor, optional
            Positional bias tensor for attention, by default None. If provided, it is expected to be of shape compatible with the attention mechanism, e.g. (batch_size, sequence_length, sequence_length)
        attn_mask : torch.Tensor, optional
            Attention mask tensor, by default None

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor] | torch.Tensor
            If `return_attn` is True, returns a tuple containing the output tensor and the attention weights tensor. Otherwise, returns only the output tensor.
        """
        x_new, attn = self.attn_layer(x, x, x, pos_bias, attn_mask)

        if self.batchnorm:
            x = (x + self.dropout(x_new)).transpose(1, 2)
            x = self.norm1(x).transpose(1, 2)
        else:
            x = self.norm1(x + self.dropout(x_new))

        x = x + self.dropout(self.feedforward(x))
        if self.batchnorm:
            x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm2(x)

        if self.return_attn:
            return x, attn
        else:
            return x


class VanillaEncoderLayer(BaseEncoderLayer):
    """
    Implementation of a vanilla encoder layer with a feedforward network.
    """

    def _build_feedforward(self, d_model: int, **ff_kwargs) -> nn.Module:
        """
        Build the feedforward network for the encoder layer.

        Parameters
        ----------
        d_model : int
            The dimension of the model.
        ff_kwargs : dict
            Additional keyword arguments for the feedforward network.

        Returns
        -------
        nn.Module
            The feedforward network module.
        """
        d_ff = ff_kwargs.get("d_ff", 2048)
        activation = ff_kwargs.get("activation", nn.ReLU)
        return nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation(),
            nn.Linear(d_ff, d_model),
        )


class Encoder(nn.Module):
    """
    A container module that stacks multiple encoder layers.
    """

    def __init__(self, enc_layers: list[BaseEncoderLayer], return_attn: bool = True):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList(enc_layers)
        self.return_attn = return_attn

    def forward(
        self, x, pos_bias=None, attn_mask=None
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | torch.Tensor:
        """
        Forward pass through the encoder stack.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the encoder, typically of shape (batch_size, sequence_length, d_model).
        pos_bias : torch.Tensor, optional
            Positional bias tensor for attention, by default None. If provided, it is expected to be of shape compatible with the attention mechanism, e.g. (batch_size, sequence_length, sequence_length)
        attn_mask : torch.Tensor, optional
            Attention mask tensor, by default None

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor]] | torch.Tensor
            If `return_attn` is True, returns a tuple containing the output tensor and a list of attention tensors from each encoder layer. Otherwise, returns only the output tensor.
        """
        attns = []
        for enc in self.encoders:
            x, attn = enc(x, pos_bias, attn_mask)
            attns.append(attn)

        if self.return_attn:
            return x, attns

        return x
