import torch
from typing import Protocol


class MaskProtocol(Protocol):
    """
    Protocol for mask classes.
    """

    mask: torch.Tensor


class TriangularCausalMask:
    """
    Triangular causal mask for sequence modeling.
    """

    def __init__(self, B: int, L: int, device: str = "cpu"):
        """
        Initialize the triangular causal mask.

        Parameters
        ----------
        B : int
            Batch size.
        L : int
            Sequence length.
        device : str, optional
            Device to store the mask on, by default "cpu".
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        """Return the triangular causal mask."""
        return self._mask
