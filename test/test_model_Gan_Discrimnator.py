import pytest
from model.Gan.Discriminator import Discriminator
import torch


# Fixture for setting up common test requirements
@pytest.fixture(scope="function")
def setup_discriminator(request):
    test_params = request.param

    input_channels = test_params['input_channels']
    batch = test_params['batch']
    input_size = test_params['input_size']

    # 網路定義
    config = {
        'in_channels': input_channels,  # 輸入通道數
        'out_channels': 1,  # 最後一層輸出通道數
        'layer_depth': test_params['layer_depth'],  # 每層的輸出通道數量
        'layer_kernel_size': test_params['layer_kernel_size'],  # 卷積核尺寸
        'layer_stride': test_params['layer_stride'],  # 卷積步長
        'layer_padding': test_params['layer_padding'],  # 卷積填充數
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化 Discriminator
    discriminator = Discriminator(config).to(device)

    # 生成假批次，輸入維度為 (batch_size, channels, height, width)
    z = torch.randn(batch, input_channels, input_size, input_size, device=device)
    output = discriminator(z)

    return output, batch


test_cases = [
    {   # 'input_size': 128,
        'input_channels': 3,
        'batch': 32,
        'input_size': 128,
        'layer_depth': [16, 32, 64, 128, 256, 512, 1024],
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4],
        'layer_stride': [2, 2, 2, 2, 2, 2, 2],
        'layer_padding': [1, 1, 1, 1, 1, 1, 1]
    },
    {   # 'input_size': 256,
        'input_channels': 1,
        'batch': 64,
        'input_size': 256,
        'layer_depth': [8, 16, 32, 64, 128, 256, 512, 1024],
        'layer_kernel_size': [4, 4, 4, 4, 4, 4, 4, 4],
        'layer_stride': [2, 2, 2, 2, 2, 2, 2, 2],
        'layer_padding': [1, 1, 1, 1, 1, 1, 1, 1]
    },
    {   # 'input_size': 64,
        'input_channels': 1,
        'batch': 87,
        'input_size': 64,
        'layer_depth': [32, 64, 128, 256, 512, 1024],
        'layer_kernel_size': [4, 4, 4, 4, 4, 4],
        'layer_stride': [2, 2, 2, 2, 2, 2],
        'layer_padding': [1, 1, 1, 1, 1, 1]
    }
]

@pytest.mark.parametrize('setup_discriminator', test_cases, indirect=True)
def test_discriminator(setup_discriminator):
    output, batch = setup_discriminator
    # batch dims
    assert output.dim() == 4
    # batch size check
    assert output.size()[0] == batch
    # channel[1], width[2], height[3]
    assert output.size()[1] == 1
    assert output.size()[2] == 1
    assert output.size()[3] == 1