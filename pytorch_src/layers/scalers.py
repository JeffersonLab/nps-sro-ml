import torch
from typing import Literal, Tuple
from base.scaler import BaseScaler


class MinMaxScaler(BaseScaler):
    """
    Implementation of Min-Max scaling for tensors in the same way as in sklearn.
    """

    def __init__(
        self,
        mode: Literal["feature", "global"] = "feature",
        feature_range: Tuple[float, float] = (0, 1),
    ):
        """
        Initialize the MinMaxScaler.

        Parameters
        ----------
        mode : Literal["feature", "global"], default="feature"
            Mode of scaling. "feature" scales each feature independently,
            while "global" scales using the global min and max across all features.
        feature_range : Tuple[float, float], default=(0, 1)
            Desired range of transformed data.
        """
        super().__init__()
        assert mode in ("feature", "global"), "mode must be 'feature' or 'global'"
        self.mode = mode
        self.feature_range = feature_range

        # avoid gradient tracking
        self.register_buffer("data_min_", None)
        self.register_buffer("data_max_", None)
        self.register_buffer("scale_", None)
        self.register_buffer("min_shift_", None)

    def fit(self, tensor: torch.Tensor) -> "MinMaxScaler":
        """
        Fit the scaler to the data. This computes and stores `data_min_`, `data_max_`, `scale_`, and `min_shift_`.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to fit the scaler.

        Returns
        -------
        MinMaxScaler
            Fitted scaler instance.
        """
        if self.mode == "feature":
            dims = list(range(tensor.dim() - 1))
            data_min = tensor.amin(dim=dims, keepdim=True)
            data_max = tensor.amax(dim=dims, keepdim=True)
        else:
            data_min = tensor.min()
            data_max = tensor.max()
            data_min = data_min.view(1, *([1] * (tensor.dim() - 1)))
            data_max = data_max.view(1, *([1] * (tensor.dim() - 1)))

        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0

        scale = (self.feature_range[1] - self.feature_range[0]) / data_range
        min_shift = self.feature_range[0] - data_min * scale

        self.data_min_ = data_min
        self.data_max_ = data_max
        self.scale_ = scale
        self.min_shift_ = min_shift
        return self

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Transform the input tensor using the fitted scaler parameters.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to be transformed.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """
        if self.scale_ is None:
            raise RuntimeError("Must fit scaler before transform()")
        return tensor * self.scale_ + self.min_shift_

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform the scaled tensor back to the original scale.

        Parameters
        ----------
        tensor : torch.Tensor
            Scaled tensor to be inverse transformed.

        Returns
        -------
        torch.Tensor
            Tensor transformed back to the original scale.
        """
        if self.scale_ is None:
            raise RuntimeError("Must fit scaler before inverse_transform()")
        return (tensor - self.min_shift_) / self.scale_
