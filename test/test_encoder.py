import pytest
import torch
from model.Autoencoder.model import ResNetEncoder

test_cases_pair = [
    (
        {
            'n_ResidualBlock': 2,
            'n_levels': 4,
            'input_ch': 3,
            'z_dim': 10,
            'bUseMultiResSkips': True
        },
        {# batch, c, w, h
            'test_input_batch': (32, 3, 256, 256),
            'out_dims_shape': (32, 10, 16, 16)
        }
    ),
]


@pytest.mark.parametrize('test_input,expected', test_cases_pair)
def test_eval(test_input, expected):
    config = test_input
    input_shape = expected['test_input_batch']
    output = expected['out_dims_shape']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    encoder = ResNetEncoder(config).to(device)
    #
    input_batch = torch.randn(input_shape).to(device)
    #
    net_output = encoder(input_batch)
    #
    assert net_output.shape == output

