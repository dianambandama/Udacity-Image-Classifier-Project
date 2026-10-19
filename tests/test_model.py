"""Small, dataset-free tests for the training and prediction interface."""
import pytest
import torch

from train import build_model, evaluate
from predict import predict


@pytest.mark.parametrize("architecture", ["vgg16", "densenet121"])
def test_classifier_output_shape(architecture):
    model = build_model(architecture, hidden_units=16, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 102)
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(1), atol=1e-5)


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError):
        build_model("unknown", pretrained=False)


def test_invalid_top_k_rejected():
    class Dummy:
        class_to_idx = {"1": 0}
    with pytest.raises(ValueError):
        predict("unused.jpg", Dummy(), torch.device("cpu"), top_k=2)
