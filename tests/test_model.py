from tests import _PATH_DATA, _PROJECT_ROOT
import torch
from ML_project.models.model import MyNeuralNet
import pytest
"""

def test_model():
    model = MyNeuralNet(in_features=4,out_features=10)
    with pytest.raises(ValueError, match = "Expected 4D a tensor"):
        model(torch.randn(1,2,3))

"""

