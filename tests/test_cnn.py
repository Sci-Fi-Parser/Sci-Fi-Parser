from src.sci_fi_parser.classifier import cnn
import pytest
import torch

def test_cnn_output_shape():
    model = cnn.CNNClassifier(
        num_classes = 5, channels_in = 3, channels_out = [16], conv_kernel_size = 3, 
        pool_size = 2, linear_layer_neurons=128, input_image_size = 64
    )
    test_in = torch.randn(16, 3, 64, 64)
    test_out = model(test_in)
    assert test_out.shape == (16, 5)

def test_cnn_output_sums_to_unity():
    model = cnn.CNNClassifier(
        num_classes = 5, channels_in = 3, channels_out = [16, 32], conv_kernel_size = 3, 
        pool_size = 2, linear_layer_neurons=128, input_image_size = 64
    )
    test_in = torch.randn(16, 3, 64, 64)
    with torch.no_grad():
        test_out = model(test_in)
        sums = torch.sum(test_out, dim=1)
        target = torch.ones(sums.shape)
        assert max(abs(target-sums))<10**-5