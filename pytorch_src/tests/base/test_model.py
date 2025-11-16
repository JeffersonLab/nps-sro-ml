import torch
import pytest
from base.model import BaseModel


import torch.nn as nn


class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.frozen_layer = nn.Linear(5, 3)
        # Freeze one layer to test parameter counting
        for param in self.frozen_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.frozen_layer(x)
        return x


class TestBaseModel:

    def test_forward_not_implemented(self):
        """Test that forward method raises NotImplementedError when not overridden"""

        class IncompleteModel(BaseModel):
            pass

        with pytest.raises(NotImplementedError):
            IncompleteModel().forward()
            

    def test_concrete_model_forward(self):
        """Test that concrete model with forward implementation works"""
        model = ConcreteModel()
        x = torch.randn(2, 10)
        output = model(x)
        assert output.shape == (2, 3)

    def test_str_includes_trainable_parameters(self):
        """Test that __str__ method includes trainable parameter count"""
        model = ConcreteModel()
        str_output = str(model)
        assert "Trainable parameters:" in str_output

    def test_str_counts_only_trainable_parameters(self):
        """Test that __str__ counts only trainable parameters (excludes frozen)"""
        model = ConcreteModel()
        str_output = str(model)

        # Calculate expected trainable parameters
        # fc1: 10*20 + 20 = 220
        # fc2: 20*5 + 5 = 105
        # frozen_layer should not be counted: 5*3 + 3 = 18
        expected_params = 220 + 105  # = 325

        assert f"Trainable parameters: {expected_params}" in str_output

    def test_str_includes_model_architecture(self):
        """Test that __str__ includes parent class string representation"""
        model = ConcreteModel()
        str_output = str(model)
        assert "ConcreteModel" in str_output
        assert "Linear" in str_output

    def test_model_inherits_from_nn_module(self):
        """Test that BaseModel inherits from nn.Module"""
        model = ConcreteModel()
        assert isinstance(model, nn.Module)

    def test_empty_model_has_zero_parameters(self):
        """Test that model with no parameters shows 0 trainable parameters"""

        class EmptyModel(BaseModel):
            def forward(self, x):
                return x

        model = EmptyModel()
        str_output = str(model)
        assert "Trainable parameters: 0" in str_output
