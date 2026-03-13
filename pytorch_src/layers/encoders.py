import torch
import torch.nn as nn
from layers.attention import AttentionLayer
from typing import Optional, Tuple, List


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
        attn_mask: torch.Tensor,
        pos_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the encoder layer, typically of shape (batch_size, sequence_length, d_model).
        attn_mask : torch.Tensor
            Attention mask tensor.
        pos_bias : torch.Tensor
            Positional bias tensor for attention.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the output tensor and the attention weights from the attention layer. The output tensor is typically of shape (batch_size, sequence_length, d_model), and the attention weights depend on the specific attention implementation.
        """
        x_new, attn = self.attn_layer(x, x, x, attn_mask, pos_bias)

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

        return x, attn


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

    def __init__(self, enc_layers: List[BaseEncoderLayer]):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList(enc_layers)

    def forward(
        self,
        x,
        attn_mask: torch.Tensor,
        pos_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the encoder stack.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the encoder, typically of shape (batch_size, sequence_length, d_model).

        attn_mask : torch.Tensor
            Attention mask tensor.

        pos_bias : torch.Tensor, optional
            Positional bias tensor for attention, by default None. If provided, it is expected to be of shape compatible with the attention mechanism, e.g. (batch_size, sequence_length, sequence_length)

        Returns
        -------
        Tuple[torch.Tensor, List[torch.Tensor]]
            A tuple containing the output tensor and the attention weights produced by the attention layer. The output tensor is typically of shape (batch_size, sequence_length, d_model), and the attention weights depend on the specific attention implementation.
        """
        attns = []
        for enc in self.encoders:
            x, attn = enc(
                x,
                attn_mask,
                pos_bias=pos_bias,
            )
            attns.append(attn)

        return x, attns
