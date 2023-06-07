import pytest
import torch
import sys
sys.path.append('./../../')
from model.Autoencoder.model import ResNetAE_Conductor

test_cases_pair = [
    (
        {
            'input_shape': (256, 256, 3),
            'n_ResidualBlock': 8,
            'n_levels': 4,
            'z_dim': 10,
            'bottleneck_dim': 128,
            'bUseMultiResSkips': True
        },
        {   #                 batch, c,   w,   h
            'test_input_batch': (10, 3, 256, 256),
            'test_output_shape': (10, 3, 256, 256)
        }
    ),
]

@pytest.mark.parametrize('test_input,expected', test_cases_pair)
def test_ae_inputShape_outputShape(test_input, expected):
    config = test_input
    input_shape = expected['test_input_batch']
    output = expected['test_output_shape']
    #
    ae = ResNetAE_Conductor(config)
    #
    input_batch = torch.randn(input_shape)
    #
    net_output = ae(input_batch)
    #
    assert net_output.shape == output