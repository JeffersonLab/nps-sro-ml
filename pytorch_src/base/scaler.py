import torch
from abc import abstractmethod


class BaseScaler(torch.nn.Module):
    """
    A base class for scalers used to normalize and denormalize data. Subclasses
    should implement the `fit`, `transform`, and `inverse_transform` methods.
    """

    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor):
        return self.transform(tensor)

    def backward(self, tensor: torch.Tensor):
        return self.inverse_transform(tensor)

    @abstractmethod
    def fit(self, tensor: torch.Tensor) -> "BaseScaler":
        """
        Fit the scaler to the data.

        Parameters
        ----------
        tensor : torch.Tensor
            The data to fit the scaler on.
        Returns
        -------
        self : BaseScaler
            The fitted scaler.
        """

    @abstractmethod
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Transform the data using the fitted scaler.

        Parameters
        ----------
        tensor : torch.Tensor
            The data to transform.
        Returns
        -------
        torch.Tensor
            The transformed data.
        """

    @abstractmethod
    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform the data using the fitted scaler.

        Parameters
        ----------
        tensor : torch.Tensor
            The data to inverse transform.
        Returns
        -------
        torch.Tensor
            The inverse transformed data.
        """
        raise NotImplementedError
