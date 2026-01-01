import math
import torch


class PositionalEmbedding(torch.nn.Module):
    """
    Implements the sinusoidal positional encoding as described in "Attention is All You Need".
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional embedding module.

        Parameters
        ----------
        d_model : int
            The dimension of the model.
        max_len : int, optional
            The maximum length of the input sequences, by default 5000.
        """
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Tensor with positional encodings added, of the same shape as input x.
        """
        if x.dim() == 2:
            x = x.unsqueeze(2)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]

        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {x.size(1)} exceeds maximum length {self.pe.size(1)}"
            )

        return x + self.pe[:, : x.size(1), :]
