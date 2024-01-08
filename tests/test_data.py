
import torch
from tests import _PATH_DATA
import pytest
import os.path

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + "/train_data.pt"), reason = "Data files not found")
def test_data():
    data = torch.load(_PATH_DATA + "/train_data.pt")
    assert data.size() == torch.Size([25000,1,28,28])

@pytest.mark.parametrize("x,y",[(1,2),(3,6),(5,10)])
def test_data_something(x,y):
    assert 2*x == y, "Check that y is twice as big as x"
